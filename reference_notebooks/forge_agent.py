%%writefile /kaggle/working/my_agent.py
"""Mixed Agent v3: Graph Explorer + Pattern Analysis + Action Planning."""
import hashlib, logging, random, time
import numpy as np
from collections import deque
from typing import Dict, Hashable, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from agents.agent import Agent
from agents.tracing import trace_agent_session
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger()

INFINITY = np.iinfo(np.int32).max
edge_dtype = np.dtype([("group","i4"),("result","i4"),("target","U32"),("distance","i4"),("errors","i4")])

# ===================== GraphExplorer =====================
@dataclass
class NodeInfo:
    name: Hashable
    total_candidates: int
    num_groups: int = 1
    active_group: int = 0
    group2remaining_candidate_ids: List[Set[int]] = field(default_factory=list)
    edge_data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=edge_dtype))
    error_threshold: int = 3
    closed: bool = False
    distance: float | None = 0

    def __post_init__(self):
        if self.num_groups == 1 and not self.group2remaining_candidate_ids:
            self.group2remaining_candidate_ids = [set(range(self.total_candidates))]
        self.group2remaining_candidate_ids = [set(r) for r in self.group2remaining_candidate_ids]
        self.edge_data = np.zeros(self.total_candidates, dtype=edge_dtype)
        for gid, rids in enumerate(self.group2remaining_candidate_ids):
            self.edge_data["group"][list(rids)] = gid

    @property
    def has_open(self):
        return len(np.where(self.edge_data["result"]==0)[0]) > 0

    def record_test(self, edge_idx, success, target_node=None):
        egid = self.edge_data["group"][edge_idx]
        assert self.edge_data["result"][edge_idx] == 0
        if success == -1:
            self.edge_data["errors"][edge_idx] += 1
            if self.edge_data["errors"][edge_idx] >= self.error_threshold:
                self.edge_data["errors"][edge_idx] = 0
                ngid = egid + 1
                if ngid > self.num_groups - 1:
                    self.group2remaining_candidate_ids[egid].discard(edge_idx)
                    self.edge_data["result"][edge_idx] = -1
                    self.edge_data["distance"][edge_idx] = INFINITY
                    return True
                self.edge_data["group"][edge_idx] = ngid
                self.group2remaining_candidate_ids[ngid].add(edge_idx)
                self.group2remaining_candidate_ids[egid].discard(edge_idx)
            return False
        self.group2remaining_candidate_ids[egid].discard(edge_idx)
        if success == 1:
            self.edge_data["target"][edge_idx] = str(target_node)
            self.edge_data["distance"][edge_idx] = -1
            self.edge_data["result"][edge_idx] = 1
        elif success == 0:
            self.edge_data["distance"][edge_idx] = INFINITY
            self.edge_data["result"][edge_idx] = -1
        return True

    def has_open_group(self, group_id):
        for i in range(group_id+1):
            if i < len(self.group2remaining_candidate_ids) and self.group2remaining_candidate_ids[i]:
                return True
        return False


class GraphExplorer:
    def __init__(self, n_groups=1):
        self._n_groups = max(1, n_groups)
        self.reset()

    def reset(self):
        self._nodes = {}
        self._G = defaultdict(set)
        self._G_rev = defaultdict(set)
        self._frontier = set()
        self._dist = {}
        self._next = {}
        self._active_group = 0
        self._suspicious = {}
        self._susp_thresh = 3
        self._empty = True

    def initialize(self, start_node=None, num_candidates=None, group2remaining_candidate_ids=None):
        if start_node is not None:
            self._add(start_node, num_candidates, group2remaining_candidate_ids)

    def record_test(self, node, edge_idx, success, target_node=None,
                    target_num_candidates=None, group2remaining_candidate_ids=None,
                    suspicious_transition=False):
        if node not in self._nodes:
            raise KeyError(node)
        ni = self._nodes[node]
        if ni.closed:
            pt = ni.edge_data["target"][edge_idx]
            if target_node == pt: return
            if self._dist.get(target_node,0) >= self._dist.get(pt,INFINITY): return
        if suspicious_transition:
            key = (node, edge_idx, target_node)
            self._suspicious[key] = self._suspicious.get(key,0)+1
            if self._suspicious[key] < self._susp_thresh: return
        ni.record_test(edge_idx, success, target_node)
        if success == 1:
            if target_node is None: raise ValueError("target_node required")
            if target_node not in self._nodes:
                if target_num_candidates is None: raise ValueError("need num_candidates")
                self._add(target_node, target_num_candidates, group2remaining_candidate_ids)
            self._G[node].add((edge_idx, target_node))
            self._G_rev[target_node].add((edge_idx, node))
            if not ni.has_open_group(self.active_group):
                self._close(node)
            if self._nodes[target_node].has_open_group(self.active_group):
                self._rebuild()
            else:
                self._close(target_node)
                self._advance(target_node)
        else:
            if not ni.has_open_group(self.active_group):
                self._close(node)
                self._advance(node)

    @property
    def active_group(self):
        return self._active_group

    def _add(self, node, nc, g2r=None):
        self._nodes[node] = NodeInfo(node, nc, self._n_groups, g2r)
        self._G[node] = set()
        self._G_rev[node] = set()
        if self._empty: self._empty = False
        if self._nodes[node].has_open_group(self.active_group):
            self._frontier.add(node)
        else:
            self._close(node); self._advance(node)

    def _close(self, node):
        ni = self._nodes[node]
        if ni.closed: return
        ni.closed = True
        self._frontier.discard(node)
        self._rebuild()

    def _rebuild(self):
        self._dist.clear(); self._next.clear()
        dq = deque(self._frontier)
        for n, ni in self._nodes.items():
            ni.distance = INFINITY; self._dist[n] = INFINITY
        for s in self._frontier:
            self._nodes[s].distance = 0; self._dist[s] = 0
        while dq:
            v = dq.popleft()
            vd = self._dist.get(v, INFINITY)
            for ei, u in self._G_rev.get(v, ()):
                ud = self._dist.get(u, INFINITY)
                self._nodes[u].edge_data["distance"][ei] = vd+1
                if ud > self._nodes[u].edge_data["distance"][ei]:
                    self._nodes[u].distance = self._nodes[u].edge_data["distance"][ei]
                    self._dist[u] = self._nodes[u].distance
                    self._next[u] = (ei, v)
                    dq.append(u)

    def _advance(self, node):
        d = self._nodes[node].distance
        while d == INFINITY and self.active_group < self._n_groups-1:
            self._active_group += 1
            self._dist.clear(); self._next.clear(); self._frontier.clear()
            for n, ni in self._nodes.items():
                ni.active_group = self.active_group
                if ni.has_open_group(self.active_group):
                    self._frontier.add(n); ni.closed = False
            self._rebuild()
            d = self._dist.get(node)

    def choose_edge(self, node, return_reasoning=False):
        ni = self._nodes[node]
        if ni.has_open_group(self.active_group):
            un = []
            for g in range(self.active_group+1):
                if g < len(ni.group2remaining_candidate_ids):
                    un.extend(list(ni.group2remaining_candidate_ids[g]))
            ei = random.choice(un) if un else 0
            r = f"untested {ei} g{self.active_group}"
        else:
            lo = ni.distance
            cs = [i for i,e in enumerate(ni.edge_data) if e["distance"]<=lo and e["result"]==1 and e["group"]<=self.active_group]
            if cs: ei = random.choice(cs); r = f"edge {ei} d{lo}"
            else:
                un = []
                for g in range(self.active_group+1):
                    if g < len(ni.group2remaining_candidate_ids):
                        un.extend(list(ni.group2remaining_candidate_ids[g]))
                ei = random.choice(un) if un else 0; r = f"fallback {ei}"
        if return_reasoning: return ei, r
        return ei


# ===================== PatternAnalyzer =====================
class PatternAnalyzer:
    BG = {0,1,2,3,4,5}
    SAL = {6,7,8,9,10,11,12,13,14,15}

    def __init__(self):
        self.prev = None; self.player_pos = None; self.mh = []

    def analyze(self, frame, sb_mask, sf, segs):
        clean = frame.copy(); clean[sb_mask] = 0
        a = {"player": [], "goals": [], "small": []}
        for sid, seg in enumerate(segs):
            c = seg["color"]; ar = seg["area"]; bb = seg["bounding_box"]
            cx, cy = (bb[0]+bb[2])//2, (bb[1]+bb[3])//2
            if c in self.SAL and 4 <= ar <= 400:
                a["small"].append({"sid": sid, "color": c, "area": ar, "center": (cx,cy), "twins": seg["number_of_twins"]})
        a["player"] = sorted([o for o in a["small"] if o["twins"]==0 and o["area"]<=100], key=lambda x: x["area"])[:3]
        a["goals"] = [o for o in a["small"] if o["twins"]>=1 or o["color"] in {11,12,14}]
        if self.prev is not None:
            diff = clean != self.prev
            ch = np.argwhere(diff)
            if len(ch) > 0 and len(ch) < 500:
                self.player_pos = tuple(ch.mean(axis=0).astype(int))
                self.mh.append(self.player_pos)
        self.prev = clean.copy()
        return a

    def reset(self):
        self.prev = None; self.player_pos = None; self.mh = []


# ===================== ActionPlanner =====================
class ActionPlanner:
    def __init__(self):
        self.strategy = "explore"; self.stuck = 0; self.thresh = 50

    def plan(self, analysis, ge, hf, na, nc, ag, avail, frames):
        if self.stuck > self.thresh:
            ss = ["explore","approach","interact"]
            self.strategy = ss[(ss.index(self.strategy)+1)%3]
            self.stuck = 0
        if self.strategy == "approach":
            r = self._approach(analysis, na, nc, ag)
            if r is not None: return r, "approach"
        if self.strategy == "interact":
            r = self._interact(analysis, nc, ag)
            if r is not None: return r, "interact"
        ei, rs = ge.choose_edge(hf, return_reasoning=True)
        return ei, f"explore:{rs}"

    def _approach(self, a, na, nc, ag):
        ps = a.get("player",[]); gs = a.get("goals",[])
        if not ps or not gs: self.stuck += 1; return None
        pp = ps[0]["center"]
        gp = min(gs, key=lambda g: abs(g["center"][0]-pp[0])+abs(g["center"][1]-pp[1]))["center"]
        dx, dy = gp[0]-pp[0], gp[1]-pp[1]
        ai = 1 if abs(dx)>abs(dy) else 0
        aid = nc + ai
        return aid if aid < na else None

    def _interact(self, a, nc, ag):
        if ag and ag[0]:
            t = min(ag[0])
            return t if t < nc else None
        return None

    def record(self, trans, score_ch):
        if trans or score_ch: self.stuck = 0
        else: self.stuck += 1

    def reset(self):
        self.strategy = "explore"; self.stuck = 0


# ===================== FrameProcessor =====================
class FrameProcessor:
    def __init__(self):
        self.sb_dist = 3; self.sb_ratio = 5; self.sb_twins = 3
        self.shape = (64,64); self.sal = {6,7,8,9,10,11,12,13,14,15}

    def segment(self, frame):
        h, w = frame.shape
        lm = np.zeros((h,w), dtype=int)-1; comps = []; cid = -1
        for y in range(h):
            for x in range(w):
                if lm[y,x] != -1: continue
                cid += 1; col = int(frame[y,x]); q = deque([(y,x)]); lm[y,x] = cid
                mnx=mxx=x; mny=mxy=y; ar = 0
                while q:
                    cy,cx = q.popleft(); ar += 1
                    mnx,mxx = min(mnx,cx),max(mxx,cx); mny,mxy = min(mny,cy),max(mxy,cy)
                    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                        ny,nx = cy+dy,cx+dx
                        if 0<=ny<h and 0<=nx<w and lm[ny,nx]==-1 and frame[ny,nx]==col:
                            lm[ny,nx]=cid; q.append((ny,nx))
                ra = (mxx-mnx+1)*(mxy-mny+1)
                comps.append(dict(bounding_box=(mnx,mny,mxx,mxy), color=col, area=ar, is_rectangle=(ar==ra)))
        for i,c in enumerate(comps):
            ts = [j for j,o in enumerate(comps) if i!=j and o["area"]==c["area"] and o["is_rectangle"]==c["is_rectangle"] and o["color"]==c["color"]]
            c["number_of_twins"] = len(ts); c["twin_ids"] = ts
        return lm, comps

    def status_bars(self, lm, segs):
        checked = set(); sbl = []
        for i,seg in enumerate(segs):
            if i in checked: continue
            checked.add(i); ids = [i]
            e = self._edge(seg)
            if not e: continue
            dirs = []
            if 'left' in e or 'right' in e: dirs.append('vertical')
            if 'top' in e or 'bottom' in e: dirs.append('horizontal')
            d = 'any' if len(dirs)==2 else (dirs[0] if dirs else 'any')
            if not self._ratio(seg, d):
                te = [t for t in seg["twin_ids"] if self._edge(segs[t])]
                for t in te: checked.add(t)
                if len(te)+1 < self.sb_twins: continue
                ids.extend(te)
            sbl.append(ids)
        mask = np.zeros(lm.shape, dtype=bool)
        for sids in sbl:
            for sid in sids: mask[lm==sid] = 1
        return sbl, mask

    def _edge(self, seg):
        x1,y1,x2,y2 = seg["bounding_box"]; r = []
        if max(x1,x2) < self.sb_dist: r.append('left')
        if min(x1,x2) > self.shape[1]-self.sb_dist: r.append('right')
        if max(y1,y2) < self.sb_dist: r.append('top')
        if min(y1,y2) > self.shape[0]-self.sb_dist: r.append('bottom')
        return r

    def _ratio(self, seg, d='any'):
        xw = seg["bounding_box"][2]-seg["bounding_box"][0]+1
        yw = seg["bounding_box"][3]-seg["bounding_box"][1]+1
        r = xw/yw
        if r>=self.sb_ratio and d in ('any','horizontal'): return True
        if r<=1/self.sb_ratio and d in ('any','vertical'): return True
        return False

    def hash(self, frame):
        frame = np.asarray(frame, dtype=np.uint8, order='C')
        flat = frame.ravel()
        if flat.size & 1: flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
        packed = (flat[0::2]<<4)|(flat[1::2]&0x0F)
        return hashlib.blake2b(packed.tobytes(), digest_size=16, person=frame.shape.__repr__().encode()).hexdigest()

    def action_groups(self, segs, ng=5):
        gs = [set() for _ in range(ng)]
        for sid,seg in enumerate(segs):
            xw = seg["bounding_box"][2]-seg["bounding_box"][0]+1
            yw = seg["bounding_box"][3]-seg["bounding_box"][1]+1
            is_s = seg["color"] in self.sal
            is_m = 2<=xw<=32 and 2<=yw<=32
            if is_s and is_m: gs[0].add(sid)
            elif is_m: gs[1].add(sid)
            elif is_s: gs[2].add(sid)
            elif seg["color"] != 16: gs[3].add(sid)
            else: gs[4].add(sid)
        return gs


# ===================== MixedAgent =====================
class MixedAgent(Agent):
    MAX_ACTIONS = 1000000
    N_GROUPS = 5
    TOTAL_TIME = 7.9*60*60
    SA = {1:GameAction.ACTION1, 2:GameAction.ACTION2, 3:GameAction.ACTION3, 4:GameAction.ACTION4, 5:GameAction.ACTION5}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        random.seed(int(time.time()*1e6)+hash(self.game_id)%1000000)
        self.fp = FrameProcessor(); self.pa = PatternAnalyzer(); self.ap = ActionPlanner()
        self.ge = GraphExplorer(n_groups=self.N_GROUPS)
        self.sbm = None; self.h2r = {}; self.h2t = {}
        self.lh = None; self.la = None; self.lao = GameAction.RESET
        self.lvhash = None; self.failed = False; self.lu = True; self.lsus = False
        self.t0 = time.time(); self.lt = time.time(); self.mst = 0.31; self.ps = 0

    @property
    def name(self):
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames, lf):
        return lf.state is GameState.WIN

    def choose_action(self, frames, lf):
        dt = time.time()-self.lt
        if dt < self.mst: time.sleep(self.mst-dt)
        self.lt = time.time()

        if lf.state in [GameState.NOT_PLAYED]:
            self.lh=None; self.la=None
            if self.failed: self.lu=True; self.failed=False
            return GameAction.RESET

        if lf.state in [GameState.GAME_OVER]:
            self.lsus=True; return GameAction.RESET

        f = np.array(lf.frame, dtype=np.uint8)
        if f.size > 0:
            nf = f.shape[0]; f = f[-1]
            if self.lu:
                sl,ss = self.fp.segment(f); _,self.sbm = self.fp.status_bars(sl,ss)
                self.h2r={}; self.h2t={}; self.pa.reset(); self.ap.reset()
            f[self.sbm]=16; sf,segs = self.fp.segment(f)
            avail = lf.available_actions; nc=0; na=0; arrows=[]
            if 6 in avail:
                na+=len(segs); nc+=len(segs)
                ag = self.fp.action_groups(segs, self.N_GROUPS)
            else:
                ag = [set() for _ in range(self.N_GROUPS)]
            for aid in avail:
                if aid in self.SA:
                    arrows.append(self.SA[aid]); ag[0].add(na); na+=1
            f[f==16]=0; hf = self.fp.hash(f)
            analysis = self.pa.analyze(f, self.sbm, sf, segs)

            if self.lu:
                self.lvhash=hf; self.ge.reset()
                self.ge.initialize(start_node=hf, num_candidates=na, group2remaining_candidate_ids=ag)

            if self.lh is not None and not self.lu:
                trans = hf != self.lh
                sus = (hf==self.lvhash and nf>1) or self.lsus; self.lsus=False
                if trans:
                    self.h2r[self.lh][self.la]=1; self.h2t[self.lh][self.la]=hf
                else:
                    self.h2r[self.lh][self.la]=-1; self.h2t[self.lh][self.la]=None
                self.ge.record_test(self.lh, self.la, trans, hf, target_num_candidates=na, group2remaining_candidate_ids=ag, suspicious_transition=sus)
                sc = lf.levels_completed if hasattr(lf,'levels_completed') else 0
                self.ap.record(trans, sc>self.ps); self.ps=sc

            if hf not in self.h2r:
                self.h2r[hf]=np.zeros(na); self.h2t[hf]=[0]*na
            if hf not in self.ge._nodes and self.lh is not None:
                try: self.ge.record_test(self.lh, self.la, trans, hf, target_num_candidates=na, group2remaining_candidate_ids=ag)
                except: pass

            aa = np.where(self.h2r[hf]!=-1)[0]
            if len(aa)==0: raise ValueError('No actions')

            action_id, reasoning = self.ap.plan(analysis, self.ge, hf, na, nc, ag, avail, frames)
            if action_id<0 or action_id>=na:
                action_id, reasoning = self.ge.choose_edge(hf, return_reasoning=True)

            if action_id < nc:
                pts = np.argwhere(sf == action_id)
                if len(pts)>0:
                    pt = pts[random.randint(0,len(pts)-1)]
                    action = GameAction.ACTION6
                    action.set_data({"x":int(pt[1]),"y":int(pt[0])})
                    action.reasoning = {"desired_action":f"{action.value}","my_reason":reasoning}
                else:
                    action = random.choice(arrows) if arrows else GameAction.RESET
            else:
                action = arrows[action_id-nc]
                action.reasoning = {"desired_action":f"{action.value}","my_reason":reasoning}

            self.lh=hf; self.la=action_id; self.lao=action
            return action

        if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET
        action = random.choice([a for a in GameAction if a is not GameAction.RESET])
        if action.is_simple(): action.reasoning = f"RNG {action.value}"
        elif action.is_complex():
            action.set_data({"x":random.randint(0,63),"y":random.randint(0,63)})
            action.reasoning = {"desired_action":f"{action.value}","my_reason":"RNG"}
        return action

    @trace_agent_session
    def main(self):
        self.timer = time.time()
        score = 0
        while not self.is_done(self.frames, self.frames[-1]) and self.action_counter <= self.MAX_ACTIONS:
            try:
                action = self.choose_action(self.frames, self.frames[-1])
            except Exception as e:
                self.failed=True; self.lu=True
                action = self.lao
            if frame := self.take_action(action):
                ns = frame.score if hasattr(frame,'score') else (frame.levels_completed if hasattr(frame,'levels_completed') else 0)
                if ns > score:
                    self.lu=True; self.sbm=None
                elif self.sbm is not None:
                    self.lu=False
                score = ns
                self.append_frame(frame)
                logger.info(f"{self.game_id} - {action.name}: count {self.action_counter}, score {score}")
            self.action_counter += 1
            if time.time()-self.t0 > self.TOTAL_TIME: break
        self.cleanup()
