"""Freeze a source-evidence review packet with shortlist origin/score hidden."""
import argparse,hashlib,json,random
from pathlib import Path

def build(lexical,diverse,papers,atoms):
    papers={p['source_id']:p for p in papers}
    atom_by_id={}
    for atom in atoms:
        h=hashlib.sha256(atom['statement'].encode()).hexdigest()
        aid='atom-'+hashlib.sha256(f"{atom['source_id']}:{h}".encode()).hexdigest()[:16]
        atom_by_id[aid]=atom
    groups={'lexical_top8':lexical['candidates'][:8],
            'diverse_top8':diverse['candidates'][:8],
            'diverse_random8':diverse['seeded_random_shortlist'][:8]}
    pool={}
    for group,rows in groups.items():
        for rank,row in enumerate(rows,1):
            key=(row['producer_atom_id'],row['target_atom_id'])
            if key not in pool:pool[key]={'row':row,'memberships':[]}
            pool[key]['memberships'].append({'group':group,'rank':rank})
    entries=list(pool.values());random.Random(84631).shuffle(entries)
    blinded=[];key=[]
    for i,entry in enumerate(entries,1):
        rid=f'C{i:03d}';row=entry['row'];record={'id':rid}
        for side in ['producer','target']:
            atom=atom_by_id[row[side+'_atom_id']];paper=papers[atom['source_id']]
            record[side]={'source_id':atom['source_id'],'url':paper['url'],'title':paper['title'],
                          'abstract':paper['abstract'],'atom_statement':atom['statement'],
                          'inputs':atom['inputs'],'outputs':atom['outputs'],'assumptions':atom.get('assumptions',[]),
                          'source_hash':atom['source_hash']}
        blinded.append(record);key.append({'id':rid,'memberships':entry['memberships'],
                                          'producer_atom_id':row['producer_atom_id'],'target_atom_id':row['target_atom_id']})
    return {'date':'2026-11-16','seed':84631,'scope':'Abstract-only machine review, shortlist origin and numerical scores hidden. Novelty and usefulness cannot be established.','items':blinded},key

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lexical',type=Path,required=True);p.add_argument('--diverse',type=Path,required=True)
    p.add_argument('--papers',type=Path,required=True);p.add_argument('--atoms',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous review packet')
    packet,key=build(json.loads(a.lexical.read_text()),json.loads(a.diverse.read_text()),json.loads(a.papers.read_text()),[json.loads(l) for l in a.atoms.read_text().splitlines() if l.strip()])
    a.output.mkdir(parents=True);(a.output/'packet.json').write_text(json.dumps(packet,indent=2)+'\n')
    (a.output/'key.json').write_text(json.dumps(key,indent=2)+'\n')
    manifest={'packet_sha256':hashlib.sha256((a.output/'packet.json').read_bytes()).hexdigest(),
              'key_sha256':hashlib.sha256((a.output/'key.json').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_sha256':{label:hashlib.sha256(getattr(a,label).read_bytes()).hexdigest() for label in ['lexical','diverse','papers','atoms']},
              'items':len(packet['items']),'scope':packet['scope']}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
