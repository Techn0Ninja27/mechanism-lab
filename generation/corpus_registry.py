"""Select a completed local evidence corpus; never select an in-progress harvest."""
import json
from pathlib import Path
SNAPSHOTS=(('corpus-arxiv-100k','bulk-manifest.json'),('corpus-integrated-v3','manifest.json'),
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
        return {'snapshot':name,'path':folder/'corpus.sqlite','records':count,
                'cutoff':manifest.get('cutoff'),'scope':'Searchable metadata and abstracts; not newly atomized papers.'}
    return None

def corpus_path(root):
    selected=active_corpus(root)
    return selected['path'] if selected else Path(root)/'results/corpus-expanded/corpus.sqlite'
