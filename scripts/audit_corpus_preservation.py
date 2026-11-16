"""Read-only preservation audit before switching evidence-corpus snapshots."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
CORE=('canonical_id','title','abstract','authors_json','published_date','updated_date','content_sha256','retrieval_method','retrieved_at','evidence_level')
INFLUENCE=('citation_count','field_weighted_citation_impact','citation_normalized_percentile','influence_provenance')

def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8_388_608),b''):h.update(block)
    return h.hexdigest()

def audit(before,after):
    with sqlite3.connect('file:'+str(after.resolve())+'?mode=ro',uri=True) as db:
        db.execute('ATTACH DATABASE ? AS old',('file:'+str(before.resolve())+'?mode=ro',))
        counts={name:db.execute(query).fetchone()[0] for name,query in {
            'before_records':'SELECT count(*) FROM old.works',
            'after_records':'SELECT count(*) FROM main.works',
            'before_aliases':'SELECT count(*) FROM old.aliases',
            'after_aliases':'SELECT count(*) FROM main.aliases',
            'fts_records':'SELECT count(*) FROM main.works_fts',
            'missing_old_sources':'SELECT count(*) FROM old.works o LEFT JOIN main.works n ON o.source_id=n.source_id WHERE n.source_id IS NULL',
            'missing_or_changed_aliases':'SELECT count(*) FROM old.aliases o LEFT JOIN main.aliases n ON o.alias=n.alias WHERE n.canonical_id IS NOT o.canonical_id',
            'fts_orphans':'SELECT count(*) FROM main.works_fts f LEFT JOIN main.works w ON w.source_id=f.source_id WHERE w.source_id IS NULL',
            'fts_duplicate_sources':'SELECT count(*) FROM (SELECT source_id FROM main.works_fts GROUP BY source_id HAVING count(*)<>1)',
            'fts_changed_text':'SELECT count(*) FROM main.works_fts f JOIN main.works w ON w.source_id=f.source_id WHERE f.title IS NOT w.title OR f.abstract IS NOT w.abstract'
        }.items()}
        changed={key:db.execute(f'SELECT count(*) FROM old.works o JOIN main.works n USING(source_id) WHERE o.{key} IS NOT n.{key}').fetchone()[0] for key in CORE}
        influence={key:db.execute('SELECT count(*) FROM old.works o JOIN main.works n USING(source_id) WHERE json_extract(o.bulk_metadata_json, ?) IS NOT NULL AND json_extract(o.bulk_metadata_json, ?) IS NOT json_extract(n.bulk_metadata_json, ?)',('$.'+key,)*3).fetchone()[0] for key in INFLUENCE}
        integrity=db.execute('PRAGMA main.quick_check').fetchall()
    old_aux=before.parent/'openai_sources.jsonl';new_aux=after.parent/'openai_sources.jsonl'
    auxiliary=old_aux.is_file() and new_aux.is_file() and file_sha(old_aux)==file_sha(new_aux)
    passed=(not counts['missing_old_sources'] and not counts['missing_or_changed_aliases'] and not counts['fts_orphans']
            and not counts['fts_duplicate_sources'] and not counts['fts_changed_text']
            and counts['fts_records']==counts['after_records'] and not any(changed.values()) and not any(influence.values())
            and integrity==[('ok',)] and auxiliary)
    return {'schema':'corpus-preservation.v1','script_sha256':file_sha(Path(__file__)),'before':str(before),'after':str(after),'counts':counts,'changed_core_fields':changed,
            'changed_existing_influence_fields':influence,'openai_auxiliary_byte_identical':auxiliary,'sqlite_quick_check':integrity,
            'passed':passed,'scope':'Preserves existing rows, aliases, content and nonmissing influence metadata; new DOI/category metadata may be added. Not corpus correctness, coverage completeness, novelty or usefulness.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous audit')
    result=audit(a.before,a.after);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not result['passed']:raise SystemExit(1)
