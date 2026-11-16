"""Select a completed local evidence corpus; never select an in-progress harvest."""
import json
from pathlib import Path
SNAPSHOTS=(('corpus-ml-complete-mirror-v6','bulk-manifest.json'),('corpus-ml-grounded-v4','bulk-manifest.json'),('corpus-openalex-enriched-v3','bulk-manifest.json'),('corpus-arxiv-100k','bulk-manifest.json'),('corpus-integrated-v3','manifest.json'),
           ('corpus-integrated-v2','manifest.json'),('corpus-expanded','manifest.json'))
def active_corpus(root):
    root=Path(root)
    for name,marker in SNAPSHOTS:
        folder=root/'results'/name
        if not (folder/'corpus.sqlite').is_file():continue
        try:manifest=json.loads((folder/marker).read_text())
        except (OSError,ValueError):continue
        count=manifest.get('records')
        if isinstance(count,bool) or not isinstance(count,int) or count<1:continue
        influence = None
        try:
            audit = json.loads((folder/'metadata-audit.json').read_text())
            if audit.get('source_sha256') == manifest.get('corpus_sha256') and audit.get('records') == count and manifest.get('corpus_sha256'):
                candidates = {key: audit['influence_fields'][key]['known'] for key in ('citation_count','citation_normalized_percentile')}
                if all(isinstance(v,int) and not isinstance(v,bool) and 0<=v<=count for v in candidates.values()): influence=candidates
        except (OSError,ValueError,KeyError,TypeError): pass
        return {'snapshot':name,'path':folder/'corpus.sqlite','records':count,'influence_coverage':influence,
                'cutoff':manifest.get('cutoff'),'scope':'Searchable metadata and abstracts; not newly atomized papers.'}
    return None

def corpus_path(root):
    selected=active_corpus(root)
    return selected['path'] if selected else Path(root)/'results/corpus-expanded/corpus.sqlite'
