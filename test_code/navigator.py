"""
Navigator - Obstacle Avoidance Navigation Module

Integrates TopoMap + Odometry for dynamic path re-planning.
Call on_obstacle() when ESP32 reports B:1.
"""

import threading


class Navigator:
    def __init__(self, topo_map, odometry=None):
        self.map = topo_map
        self.odometry = odometry
        self.lock = threading.Lock()

        self.mission_plan = []
        self.visited_patrols = set()
        self.blocked_edges = set()
        self.current_node = None
        self.current_path = None
        self.path_index = 0
        self.mission_complete = False
        self.all_patrols_visited = False

    def init_mission(self, start_node="S"):
        with self.lock:
            self.mission_plan = self.map.get_mission_plan()
            self.visited_patrols.clear()
            self.blocked_edges.clear()
            self.current_node = start_node
            self.current_path = None
            self.path_index = 0
            self.start_node = start_node
            self.mission_complete = False
            self.all_patrols_visited = False

    def mark_visited(self, patrol_id):
        with self.lock:
            node_id = f"P{patrol_id}"
            self.visited_patrols.add(node_id)
            self.current_node = node_id
            self.current_path = None
            self.path_index = 0
            if self.visited_patrols.issuperset(set(self.mission_plan)):
                self.all_patrols_visited = True

    def mark_position(self, node_id):
        with self.lock:
            self.current_node = node_id

    def on_obstacle(self, from_node, to_node):
        with self.lock:
            self.blocked_edges.add((from_node, to_node))
            self.blocked_edges.add((to_node, from_node))
            path = self._replan_to_target()
            if path:
                self.current_path = path
                self.path_index = 0
                self.current_node = path[0]
                return True
            return False

    def get_next_segment(self):
        with self.lock:
            if self.current_path and self.path_index < len(self.current_path) - 1:
                return self.current_path[self.path_index:]

            if self.all_patrols_visited:
                if self.current_node == self.start_node:
                    self.mission_complete = True
                    return None
                path = self.map.find_path(self.current_node, self.start_node, self.blocked_edges)
                if path:
                    self.current_path = path
                    self.path_index = 0
                    return path
                return None

            path = self._replan_to_target()
            if path:
                self.current_path = path
                self.path_index = 0
                return path
            return None

    def get_next_action(self):
        if self.mission_complete:
            return ("STOP", None)
        if not self.current_path or self.path_index >= len(self.current_path) - 1:
            path = self.get_next_segment()
            if not path:
                return ("WAIT", None)

        from_node = self.current_path[self.path_index]
        to_node = self.current_path[self.path_index + 1]
        edge = self.map.get_edge(from_node, to_node)
        if edge and edge.actions:
            for action in edge.actions:
                if action != self.map.ACT_FOLLOW_LINE:
                    return (action, to_node)
            return (self.map.ACT_FOLLOW_LINE, to_node)

    def advance_path(self):
        with self.lock:
            if self.current_path and self.path_index < len(self.current_path) - 1:
                self.path_index += 1
                self.current_node = self.current_path[self.path_index]

    def get_progress(self):
        with self.lock:
            visited = len(self.visited_patrols)
            total = len(self.mission_plan)
            return {
                "visited": visited,
                "total": total,
                "current": self.current_node,
                "blocked_edges": len(self.blocked_edges),
                "complete": self.mission_complete,
            }

    def _replan_to_target(self):
        for target in self.mission_plan:
            if target not in self.visited_patrols:
                path = self.map.find_path(self.current_node, target, self.blocked_edges)
                if path:
                    return path
        if self.current_node != self.start_node and self.all_patrols_visited:
            return self.map.find_path(self.current_node, self.start_node, self.blocked_edges)
        return None
