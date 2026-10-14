"""
TopoMap - 拓扑地图模块

将赛道建模为有向图，支持路径规划、绕行重规划和 RFID 导航。

用法：
    map = TopoMap()
    map.add_node("S", "START")
    map.add_node("C1", "CROSSROAD")
    map.add_edge("S", "C1", ["FOLLOW_LINE"])
    path = map.find_path("S", "E")
    actions = map.get_actions_between("C1", "C2")
"""

import json
from collections import deque


class MapNode:
    def __init__(self, node_id, node_type, rfid_id=None, position=None):
        self.id = node_id
        self.type = node_type
        self.rfid_id = rfid_id
        self.position = position

    def __repr__(self):
        return f"MapNode({self.id}, {self.type})"


class MapEdge:
    def __init__(self, from_id, to_id, actions, expected_heading=0,
                 expected_dist=0, bypass=None):
        self.from_id = from_id
        self.to_id = to_id
        self.actions = actions
        self.expected_heading = expected_heading
        self.expected_dist = expected_dist
        self.bypass = bypass or []

    def __repr__(self):
        return f"MapEdge({self.from_id} -> {self.to_id}, {self.actions})"


class TopoMap:
    NODE_START = "START"
    NODE_PATROL = "PATROL_POINT"
    NODE_CROSS = "CROSSROAD"
    NODE_TUNNEL = "TUNNEL"
    NODE_STOP = "STOP"

    ACT_FOLLOW_LINE = "FOLLOW_LINE"
    ACT_FORWARD = "FORWARD"
    ACT_LEFT = "LEFT"
    ACT_RIGHT = "RIGHT"
    ACT_U_TURN = "U_TURN"
    ACT_STOP = "STOP"
    ACT_FOLLOW_CROSS = "FOLLOW_CROSS"

    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, node_type, rfid_id=None, position=None):
        self.nodes[node_id] = MapNode(node_id, node_type, rfid_id, position)
        return self

    def add_edge(self, from_id, to_id, actions, expected_heading=0,
                 expected_dist=0, bypass=None):
        edge = MapEdge(from_id, to_id, actions,
                       expected_heading, expected_dist, bypass)
        self.edges.setdefault(from_id, []).append(edge)
        return self

    def clear(self):
        self.nodes.clear()
        self.edges.clear()

    def find_path(self, start_id, target_id, blocked_edges=None):
        if blocked_edges is None:
            blocked_edges = set()
        if start_id not in self.nodes or target_id not in self.nodes:
            return None
        visited = {start_id}
        queue = deque([(start_id, [start_id])])
        while queue:
            current, path = queue.popleft()
            if current == target_id:
                return path
            # Search both outgoing and incoming edges (undirected)
            seen_edges = set()
            for edge in self.edges.get(current, []):
                nid = edge.to_id
                if (current, nid) in seen_edges:
                    continue
                seen_edges.add((current, nid))
                if (current, nid) in blocked_edges:
                    if edge.bypass:
                        alt = self._try_bypass(current, edge.bypass, blocked_edges)
                        if alt:
                            remaining = self.find_path(alt[-1], target_id, blocked_edges)
                            if remaining:
                                return alt + remaining[1:]
                    continue
                if nid not in visited:
                    visited.add(nid)
                    queue.append((nid, path + [nid]))
            # Also check incoming edges (for undirected search)
            for fid, elist in self.edges.items():
                for edge in elist:
                    if edge.to_id == current and fid not in visited:
                        if (fid, current) in blocked_edges:
                            continue
                        if (fid, current) in seen_edges:
                            continue
                        seen_edges.add((fid, current))
                        visited.add(fid)
                        queue.append((fid, path + [fid]))
        return None

    def _try_bypass(self, start_id, bypass_route, blocked_edges):
        path = [start_id]
        current = start_id
        for nid in bypass_route:
            if (current, nid) in blocked_edges:
                return None
            path.append(nid)
            current = nid
        return path

    def get_actions_between(self, from_id, to_id):
        for edge in self.edges.get(from_id, []):
            if edge.to_id == to_id:
                return list(edge.actions)
        return None

    def get_edge(self, from_id, to_id):
        for edge in self.edges.get(from_id, []):
            if edge.to_id == to_id:
                return edge
        return None

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_node_by_rfid(self, rfid_id):
        for node in self.nodes.values():
            if node.rfid_id == rfid_id:
                return node
        return None

    def get_next_node_in_path(self, current_node_id, path):
        if current_node_id in path:
            idx = path.index(current_node_id)
            if idx + 1 < len(path):
                return path[idx + 1]
        return None

    def get_node_by_rfid_in_path(self, rfid_id, path):
        node = self.get_node_by_rfid(rfid_id)
        if node and node.id in path:
            nxt = self.get_next_node_in_path(node.id, path)
            return node.id, nxt
        return None, None

    def build_default_map(self):
        self.clear()
        self.add_node("S",  self.NODE_START)
        self.add_node("P1", self.NODE_PATROL, rfid_id=1)
        self.add_node("P2", self.NODE_PATROL, rfid_id=2)
        self.add_node("P3", self.NODE_PATROL, rfid_id=3)
        self.add_node("C1", self.NODE_CROSS)
        self.add_node("C2", self.NODE_CROSS)
        self.add_node("T1", self.NODE_TUNNEL)
        self.add_node("E",  self.NODE_STOP)
        self.add_edge("S",  "C1", [self.ACT_FOLLOW_LINE],
                      expected_heading=0)
        self.add_edge("C1", "C2", [self.ACT_LEFT, self.ACT_FOLLOW_LINE],
                      expected_heading=270,
                      bypass=["C1", "P1", "C2"])
        self.add_edge("C1", "P1", [self.ACT_FOLLOW_LINE])
        self.add_edge("P1", "C2", [self.ACT_FOLLOW_LINE])
        self.add_edge("C2", "P2", [self.ACT_LEFT, self.ACT_FOLLOW_LINE])
        self.add_edge("P2", "C2", [self.ACT_RIGHT, self.ACT_FOLLOW_LINE])
        self.add_edge("C2", "T1", [self.ACT_FOLLOW_CROSS, self.ACT_FOLLOW_LINE])
        self.add_edge("T1", "P3", [self.ACT_FOLLOW_LINE])
        self.add_edge("P3", "E",  [self.ACT_FOLLOW_LINE],
                      bypass=["P3", "C2", "E"])
        return self

    def export_json(self, path):
        data = {
            "nodes": {nid: {"type": n.type, "rfid_id": n.rfid_id,
                            "position": n.position}
                      for nid, n in self.nodes.items()},
            "edges": {fid: [{"to_id": e.to_id,
                             "actions": e.actions,
                             "expected_heading": e.expected_heading,
                             "expected_dist": e.expected_dist,
                             "bypass": e.bypass}
                            for e in elist]
                      for fid, elist in self.edges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def import_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.clear()
        for nid, nd in data["nodes"].items():
            self.add_node(nid, nd["type"], nd.get("rfid_id"), nd.get("position"))
        for fid, elist in data["edges"].items():
            for ed in elist:
                self.add_edge(fid, ed["to_id"], ed["actions"],
                              ed.get("expected_heading", 0),
                              ed.get("expected_dist", 0),
                              ed.get("bypass"))
        return self

    def pretty_print(self):
        for nid, node in self.nodes.items():
            extra = ""
            if node.rfid_id:
                extra += f" [RFID:{node.rfid_id}]"
            if node.position:
                extra += f" {node.position}"
            print(f"  [{node.type:<13}] {nid}{extra}")
        print("  ---")
        for fid, elist in self.edges.items():
            for e in elist:
                bp = f" bypass:{e.bypass}" if e.bypass else ""
                print(f"  {fid} -> {e.to_id}  {e.actions}{bp}")




    def build_competition_track(self):
        """Correct 4x4 grid map from team's analysis.
        
        12 numbered circles N1-N12, 8 patrol points P1-P8,
        extended to 12 patrol points P1-P12.
        4 tunnels T1-T4, 6 internal intersections I*_*.
        
        Grid: 4 columns (outer/inner) x 4 rows (top to bottom)
        """
        self.clear()
        
        # 12 numbered circles
        for i in range(1, 13):
            self.add_node(f"N{i}", self.NODE_CROSS)
        
        # 6 internal intersections
        for r in [1, 2, 3]:
            self.add_node(f"I1_{r}", self.NODE_CROSS)
            self.add_node(f"I12_{r}", self.NODE_CROSS)
        
        # 12 patrol points (P1-P8 from original map, P9-P12 added)
        for i in range(1, 13):
            self.add_node(f"P{i}", self.NODE_PATROL, rfid_id=i)
        
        # 4 tunnels
        for i in range(1, 5):
            self.add_node(f"T{i}", self.NODE_TUNNEL)
        
        # Start area
        self.add_node("START_AREA", self.NODE_START)
        
        # Helper for bidirectional edges
        def bidir(a, b, action):
            self.add_edge(a, b, [action])
            self.add_edge(b, a, [action])
        
        # === START: top, connected to N1 and N12 ===
        bidir("START_AREA", "N1", self.ACT_FOLLOW_LINE)
        bidir("START_AREA", "N12", self.ACT_FOLLOW_LINE)
        bidir("N1", "N12", self.ACT_FOLLOW_LINE)  # top cross
        
        # === LEFT OUTER column (N2-N3-P2-N4-N5) ===
        bidir("N2", "N3", self.ACT_FORWARD)
        bidir("N3", "P2", self.ACT_FORWARD)
        bidir("P2", "N4", self.ACT_FORWARD)
        bidir("N4", "N5", self.ACT_FORWARD)
        
        # === LEFT INNER column (N1-I1_1-P1-I1_2-I1_3-P4-N6) ===
        bidir("N1", "I1_1", self.ACT_FORWARD)
        bidir("I1_1", "P1", self.ACT_FORWARD)
        bidir("P1", "I1_2", self.ACT_FORWARD)
        bidir("I1_2", "I1_3", self.ACT_FORWARD)
        bidir("I1_3", "P4", self.ACT_FORWARD)
        bidir("P4", "N6", self.ACT_FORWARD)
        
        # === RIGHT INNER column (N12-I12_1-I12_2-P5-I12_3-N7) ===
        bidir("N12", "I12_1", self.ACT_FORWARD)
        bidir("I12_1", "I12_2", self.ACT_FORWARD)
        bidir("I12_2", "P5", self.ACT_FORWARD)
        bidir("P5", "I12_3", self.ACT_FORWARD)
        bidir("I12_3", "N7", self.ACT_FORWARD)
        
        # === RIGHT OUTER column (N11-N10-P7-N9-N8) ===
        bidir("N11", "N10", self.ACT_FORWARD)
        bidir("N10", "P7", self.ACT_FORWARD)
        bidir("P7", "N9", self.ACT_FORWARD)
        bidir("N9", "N8", self.ACT_FORWARD)
        
        # === ROW 1 (top): N2 - I1_1 - T1 - I12_1 - P6 - N11 ===
        bidir("N2", "I1_1", self.ACT_FOLLOW_LINE)
        bidir("I1_1", "T1", self.ACT_FOLLOW_LINE)
        bidir("T1", "I12_1", self.ACT_FOLLOW_LINE)
        bidir("I12_1", "P6", self.ACT_FOLLOW_LINE)
        bidir("P6", "N11", self.ACT_FOLLOW_LINE)
        
        # === ROW 2: N3 - I1_2 - T2 - I12_2 - N10 ===
        bidir("N3", "I1_2", self.ACT_FOLLOW_LINE)
        bidir("I1_2", "T2", self.ACT_FOLLOW_LINE)
        bidir("T2", "I12_2", self.ACT_FOLLOW_LINE)
        bidir("I12_2", "N10", self.ACT_FOLLOW_LINE)
        
        # === ROW 3: N4 - P3 - I1_3 - T3 - I12_3 - N9 ===
        bidir("N4", "P3", self.ACT_FOLLOW_LINE)
        bidir("P3", "I1_3", self.ACT_FOLLOW_LINE)
        bidir("I1_3", "T3", self.ACT_FOLLOW_LINE)
        bidir("T3", "I12_3", self.ACT_FOLLOW_LINE)
        bidir("I12_3", "N9", self.ACT_FOLLOW_LINE)
        
        # === ROW 4 (bottom): N5 - N6 - T4 - N7 - P8 - N8 ===
        bidir("N5", "N6", self.ACT_FOLLOW_LINE)
        bidir("N6", "T4", self.ACT_FOLLOW_LINE)
        bidir("T4", "N7", self.ACT_FOLLOW_LINE)
        bidir("N7", "P8", self.ACT_FOLLOW_LINE)
        bidir("P8", "N8", self.ACT_FOLLOW_LINE)
        
        # === PATROL POINTS P9-P12 (added for full 12) ===
        # P9: bottom row, between N6 and T4
        bidir("N6", "P9", self.ACT_FOLLOW_LINE)
        bidir("P9", "T4", self.ACT_FOLLOW_LINE)
        # P10: row 3, between N4 and P3
        bidir("N4", "P10", self.ACT_FOLLOW_LINE)
        bidir("P10", "P3", self.ACT_FOLLOW_LINE)
        # P11: row 2, between I1_2 and T2
        bidir("I1_2", "P11", self.ACT_FOLLOW_LINE)
        bidir("P11", "T2", self.ACT_FOLLOW_LINE)
        # P12: row 1, between I1_1 and T1
        bidir("I1_1", "P12", self.ACT_FOLLOW_LINE)
        bidir("P12", "T1", self.ACT_FOLLOW_LINE)
        
        return self

    def get_patrol_route(self):
        return ["START_AREA", "N1", "I1_1", "P1", "I1_2", "I1_3", "P4", "N6", "P9", "T4", "N7", "P8", "N8",
                "N9", "N10", "P7", "N9", "I12_3", "P5", "I12_2", "I12_1", "P6", "N11",
                "N10", "P7", "N9", "I12_3", "T3", "I1_3", "P3", "N4", "P2", "N3", "N2", "I1_1", "P12", "T1", "I12_1", "N12",
                "START_AREA"]

    def get_mission_plan(self):
        return [f"P{i}" for i in range(1, 13)]

    def find_route_to_next_patrol(self, current, blocked_edges=None):
        plan = self.get_mission_plan()
        if current in plan:
            idx = plan.index(current)
            if idx + 1 < len(plan):
                return self.find_path(current, plan[idx+1], blocked_edges)
        return None


if __name__ == "__main__":
    m = TopoMap()
    m.build_competition_track()
    m.pretty_print()
    print()
    path = m.find_path("START_AREA", "N8")
    print(f"Path START_AREA -> N8: {" -> ".join(path) if path else "None"}")
