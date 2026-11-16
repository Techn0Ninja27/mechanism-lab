"""Fresh task families for utility ranking; metadata never enters model features."""
import random

N=4096
FAMILIES={'train':('walk','runs','periodic','uniform'),'validation':('clipped_noise','modular_ramp','alternating_burst'),
          'heldout':('piecewise_jump','regime_switch','bitwidth_extremes')}
START={'train':20000,'validation':20200,'heldout':20300}
COUNTS={'train':120,'validation':60,'heldout':60}
LOW=-(1<<63);HIGH=(1<<63)-1

def clipped(x):return min(HIGH,max(LOW,int(x)))

def task(split,index,allow_heldout=False):
    if split not in FAMILIES or not 0<=index<COUNTS[split]:raise ValueError('invalid task')
    if split=='heldout' and not allow_heldout:raise PermissionError('heldout task blocked before freeze')
    seed=START[split]+index;r=random.Random(seed);family=FAMILIES[split][index%len(FAMILIES[split])]
    regime=(index//len(FAMILIES[split]))%4;scale=(1,16,1024,1<<25)[regime]
    offset=r.randint(-1000,1000)*scale;xs=[]
    if family=='walk':
        x=offset;drift=r.choice((-3,0,2,9))*scale
        for _ in range(N):x=clipped(x+drift+r.randint(-4,4)*scale);xs.append(x)
    elif family=='runs':
        while len(xs)<N:xs.extend([clipped(offset+r.randint(-100,100)*scale)]*r.randint(2,2**(regime+4)))
    elif family=='periodic':
        period=r.choice((7,19,61,127))
        xs=[clipped(offset+(i%period)*scale+(r.randint(-100,100)*scale if r.random()<.03 else 0)) for i in range(N)]
    elif family=='uniform':
        bits=(8,16,32,62)[regime];xs=[r.randrange(-(1<<(bits-1)),1<<(bits-1)) for _ in range(N)]
    elif family=='clipped_noise':
        xs=[clipped(offset+max(-4*scale,min(4*scale,r.gauss(0,scale)))) for _ in range(N)]
    elif family=='modular_ramp':
        modulus=r.choice((31,257,4093,65521));step=r.choice((1,3,7,23))
        xs=[clipped(offset+((i*step)%modulus)*scale) for i in range(N)]
    elif family=='alternating_burst':
        for i in range(N):xs.append(clipped(offset+(r.randint(-10000,10000)*scale if (i//64)%2 else (i%8)*scale)))
    elif family=='piecewise_jump':
        x=offset
        for _ in range(N):
            if r.random()<.015:x=clipped(r.randint(-1_000_000,1_000_000)*scale)
            else:x=clipped(x+r.randint(-2,2)*scale)
            xs.append(x)
    elif family=='regime_switch':
        while len(xs)<N:
            width=r.randint(20,180);kind=r.randrange(3);base=r.randint(-10000,10000)*scale
            xs.extend([clipped(base+(r.randint(-1000,1000)*scale if kind==0 else j*scale if kind==1 else 0)) for j in range(width)])
    else:
        choices=([0,1,-1],[0,127,-128,255,-256],[LOW,LOW+1,HIGH,HIGH-1],[0,1,-1,LOW,HIGH])[regime]
        xs=[r.choice(choices) for _ in range(N)]
    xs=xs[:N]
    assert len(xs)==N and all(LOW<=x<=HIGH for x in xs)
    return {'id':f'{split}-{index:03d}','seed':seed,'family':family,'stream_group':f'{family}:{regime}:{index}','values':xs}
