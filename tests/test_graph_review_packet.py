import hashlib,json,tempfile,unittest
from pathlib import Path
from scripts.build_graph_review_packet import build
from scripts.summarize_graph_review import summarize

class GraphReviewTests(unittest.TestCase):
    def test_packet_deduplicates_and_hides_selection_metadata(self):
        atoms=[];papers=[];ids=[]
        for name in ['a','b']:
            statement='operation '+name;h=hashlib.sha256(statement.encode()).hexdigest()
            ids.append('atom-'+hashlib.sha256(f'{name}:{h}'.encode()).hexdigest()[:16])
            atoms.append({'source_id':name,'source_hash':'hash-'+name,'statement':statement,'kind':'method','inputs':['x'],'outputs':['y']})
            papers.append({'source_id':name,'url':'https://example.test/'+name,'title':name,'abstract':'source '+name})
        row={'producer_atom_id':ids[0],'target_atom_id':ids[1],'interface_tfidf_cosine':1.0}
        lexical={'candidates':[row]};diverse={'candidates':[row],'seeded_random_shortlist':[row]}
        packet,key=build(lexical,diverse,papers,atoms)
        self.assertEqual(len(packet['items']),1);self.assertEqual(len(key[0]['memberships']),3)
        self.assertNotIn('tfidf',json.dumps(packet));self.assertNotIn('memberships',json.dumps(packet))
        self.assertNotIn('kind',packet['items'][0]['producer'])
        self.assertEqual((packet,key),build(lexical,diverse,papers,atoms))

    def test_summary_rejects_wrong_packet_and_duplicate_review_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);ids=[f'C{i:03d}' for i in range(8)]
            (p/'packet.json').write_text(json.dumps({'items':[{'id':i} for i in ids]}))
            (p/'key.json').write_text(json.dumps([{'id':i,'memberships':[{'group':'fixture','rank':n}]} for n,i in enumerate(ids)]))
            digest=hashlib.sha256((p/'packet.json').read_bytes()).hexdigest()
            (p/'manifest.json').write_text(json.dumps({'packet_sha256':digest,'key_sha256':hashlib.sha256((p/'key.json').read_bytes()).hexdigest()}))
            review={'packet_sha256':digest,'reviews':[{'id':i,'label':'unsupported_connection','confidence':'low','reason':'reason','missing_contract_or_bridge':'missing type'} for i in ids]}
            for name in ['a','b']:(p/f'reviewer-{name}.json').write_text(json.dumps(review))
            self.assertEqual(summarize(p)['exact_agreement'],8)
            review['reviews'][-1]['id']=ids[0];(p/'reviewer-b.json').write_text(json.dumps(review))
            with self.assertRaises(ValueError):summarize(p)
            review['packet_sha256']='wrong';(p/'reviewer-b.json').write_text(json.dumps(review))
            with self.assertRaises(ValueError):summarize(p)

if __name__=='__main__':unittest.main()
