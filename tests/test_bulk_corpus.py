import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.import_bulk_corpus import import_bulk, normalize_record, select_stream

class BulkCorpusTests(unittest.TestCase):
    def test_normalizes_versions_categories_and_ignores_generated_summary(self):
        row = normalize_record({"id":"2401.00001v2","title":" T ","abstract":" A ","mechanism_summary":"fake","categories":"cs.LG cs.AI","versions":[{"created":"2024-01-01"},{"created":"2024-02-01"}],"authors":[{"name":"A"}]}, source_file="x", source_file_sha256="h", cutoff="2026-11-16")
        self.assertEqual((row["canonical_id"],row["published_date"],row["updated_date"],row["categories"]),("arxiv:2401.00001","2024-01-01","2024-02-01",["cs.AI","cs.LG"]))
        self.assertNotIn("mechanism_summary", row)
    def test_existing_noncanonical_alias_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/"base";base.mkdir()
            old=base/"works.jsonl";old.write_text(json.dumps({"source_id":"local:kept","canonical_id":"local:kept","title":"Existing paper title","abstract":"A","published_date":"2020-01-01"})+'\n')
            (base/"aliases.json").write_text('{"old-version":"local:kept"}')
            source=root/"new.jsonl";source.write_text(json.dumps({"id":"2401.00001","title":"New","abstract":"A"})+'\n')
            out=root/"out";import_bulk(source,out,existing=old,max_new=1,cutoff="2026-11-16",seed=1)
            self.assertEqual(json.loads((out/"aliases.json").read_text())["old-version"],"local:kept")

    def test_influence_observation_and_arxiv_mirror_are_preserved(self):
        row=normalize_record({"id":"https://openalex.org/W8","title":"T","abstract":"A","cited_by_count":0,"retrieved_at":"2026-11-16","source_hash":"record-hash"},source_file="snapshot.jsonl",source_file_sha256="file-hash",cutoff="2026-11-16")
        obs=row["influence_provenance"][0]
        self.assertEqual(obs["as_of"],"2026-11-16")
        self.assertEqual(obs["values"]["citation_count"],0)
        row=normalize_record({"id":"2401.00001","title":"T","abstract":"A","metadata_dataset":"mirror","metadata_revision":"revision"},source_file="snapshot.jsonl",source_file_sha256="file-hash",cutoff="2026-11-16")
        self.assertEqual((row["mirror_source"],row["mirror_revision"]),("mirror","revision"))
        self.assertEqual(row["influence_provenance"],[])

    def test_openalex_no_doi_keeps_provider_url_and_query_group(self):
        row = normalize_record({"source_id":"openalex:W123", "canonical_id":"https://openalex.org/W123", "title":"T", "abstract":"A", "query_slot":"machine_learning:T1:0"}, source_file="x", source_file_sha256="h", cutoff="2026-11-16")
        self.assertEqual(row["landing_url"], "https://openalex.org/W123")
        self.assertEqual(row["corpus_group"], "ml")
        self.assertEqual(row["query_slot"], "machine_learning:T1:0")
    def test_openalex_inverted_index_and_influence_fields(self):
        row = normalize_record({"id":"https://openalex.org/W1","display_name":"T","abstract_inverted_index":{"hello":[1],"world":[0]},"publication_date":"2024-01-01","topics":[{"id":"https://openalex.org/T1"}],"cited_by_count":12,"fwci":2.1}, source_file="x", source_file_sha256="h", cutoff="2026-11-16")
        self.assertEqual(row["abstract"],"world hello"); self.assertEqual(row["citation_count"],12); self.assertEqual(row["field_weighted_citation_impact"],2.1); self.assertIn("T1",row["categories"])
    def test_accepts_collector_normalized_openalex_record(self):
        row = normalize_record({"source_id":"openalex:W1","canonical_id":"https://openalex.org/W1",
            "title":"T","abstract":"A","publication_date":"2024-01-01","cited_by_count":3,
            "fwci":1.1,"retrieved_at":"2026-11-16","retrieval_method":"openalex_api","topics":[]},
            source_file="x", source_file_sha256="h", cutoff="2026-11-16")
        self.assertEqual((row["source_id"],row["canonical_id"],row["retrieved_at"]),
                         ("openalex:W1","openalex:W1","2026-11-16"))
        self.assertEqual(row["evidence_level"],"indexed_abstract")

    def test_extracts_arxiv_location_and_merges_into_old_canonical(self):
        row = normalize_record({"source_id":"openalex:W1","canonical_id":"https://openalex.org/W1",
            "title":"A sufficiently long and stable paper title for matching",
            "abstract":"new abstract","publication_date":"2024-01-01",
            "locations":[{"landing_page_url":"https://arxiv.org/abs/2401.00001v2"}]},
            source_file="x", source_file_sha256="h", cutoff="2026-11-16")
        self.assertEqual(row["arxiv_id"],"2401.00001v2")
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/"m.jsonl"; source.write_text(json.dumps({
                "source_id":"openalex:W1","canonical_id":"https://openalex.org/W1",
                "title":"A sufficiently long and stable paper title for matching",
                "abstract":"new abstract","publication_date":"2024-01-01",
                "locations":[{"landing_page_url":"https://arxiv.org/abs/2401.00001v2"}]})+"\n")
            old=root/"old.jsonl"; old.write_text(json.dumps({
                "source_id":"arxiv:2401.00001v1","canonical_id":"arxiv:2401.00001",
                "arxiv_id":"2401.00001v1","title":"A sufficiently long and stable paper title for matching",
                "abstract":"old abstract","published_date":"2020-01-01"})+"\n")
            manifest=import_bulk(source,root/"out",existing=old,max_new=10,cutoff="2026-11-16",seed=1,groups={"ml":("cs.LG",)})
            self.assertEqual(manifest["records"],1)
            saved=json.loads((root/"out/works.jsonl").read_text().splitlines()[0])
            self.assertEqual(saved["canonical_id"],"arxiv:2401.00001")
            self.assertEqual(saved["bulk_aliases"],["openalex:W1"])
    def test_stream_selection_is_deterministic_and_quotas_cover_groups(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"m.jsonl"; rows=[{"id":f"2401.{i:05d}v1","title":f"T{i}","abstract":"A","categories":["cs.LG" if i%2 else "math.PR"],"versions":[{"created":"2024-01-01"}]} for i in range(30)]
            path.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
            a,ma=select_stream(path,max_new=10,cutoff="2026-11-16",seed=4,groups={"ml":("cs.LG",),"math":("math.",)})
            b,mb=select_stream(path,max_new=10,cutoff="2026-11-16",seed=4,groups={"ml":("cs.LG",),"math":("math.",)})
            self.assertEqual([x["source_id"] for x in a],[x["source_id"] for x in b]); self.assertEqual(ma["selected"],10); self.assertEqual(set(ma["selected_by_group"]),{"ml","math"})
    def test_import_preserves_existing_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/"m.jsonl"; source.write_text(json.dumps({"id":"2401.00001v1","title":"N","abstract":"A","categories":["cs.LG"],"versions":[{"created":"2024-01-01"}]})+"\n")
            old=root/"old.jsonl"; old.write_text(json.dumps({"source_id":"local:x","canonical_id":"local:x","title":"Old","abstract":"A","published_date":"2020-01-01"})+"\n")
            (root/"openai_sources.jsonl").write_text('{"source_id":"openai:x","title":"Source"}\n')
            out=root/"out"; m=import_bulk(source,out,existing=old,max_new=10,cutoff="2026-11-16",seed=1,groups={"ml":("cs.LG",)})
            self.assertEqual((m["existing_records"],m["bulk_selected"],m["records"]),(1,1,2))
            with self.assertRaises(FileExistsError): import_bulk(source,out,existing=old,max_new=10,cutoff="2026-11-16",seed=1,groups={"ml":("cs.LG",)})
            with sqlite3.connect(out/"corpus.sqlite") as db: self.assertEqual(db.execute("select categories_json from works where source_id='arxiv:2401.00001v1'").fetchone()[0], '["cs.LG"]')
            self.assertEqual((out/"openai_sources.jsonl").read_text(), (root/"openai_sources.jsonl").read_text())
