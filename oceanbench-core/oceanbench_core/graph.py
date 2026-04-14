from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


@dataclass
class WaypointGraph:
    """
    Waypoint graph G = (V, E) with travel-time edge costs.

    Nodes are identified by integer ids [0, ..., N-1] and carry latitude /
    longitude coordinates (in degrees). Edges are undirected by default and
    store both their great-circle length in meters and an induced travel-time
    cost in seconds.

    This class is intentionally lightweight: it provides just enough
    functionality for the Binney et al. (2013) GRG planner:

    - deterministic construction given a seed,
    - shortest-path travel time and paths,
    - path travel-time accumulation,
    - simple JSON-serializable metadata for logging.
    """

    graph: nx.Graph
    speed_mps: float

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def grid(
        cls,
        region: Mapping[str, float],
        n_lat: int,
        n_lon: int,
        *,
        n_depth: int = 1,
        speed_mps: float,
        connectivity: str = "4",
        seed: Optional[int] = None,
    ) -> "WaypointGraph":
        """
        Build a grid graph over a region, optionally including depth layers.

        Parameters
        ----------
        region:
            Mapping with keys ``lat_min``, ``lat_max``, ``lon_min``,
            ``lon_max`` (degrees).  May also include ``depth_min`` and
            ``depth_max`` (meters, positive-down) for 3-D grids.
        n_lat, n_lon:
            Number of grid points along latitude and longitude.
        n_depth:
            Number of depth layers.  When ``1`` (default) the graph is a
            standard 2-D grid and no ``depth`` attribute is stored on nodes.
        speed_mps:
            Robot speed in meters per second. Edge travel times are
            ``length_m / speed_mps``.
        connectivity:
            Grid connectivity pattern:

            - ``"4"`` — rectilinear (up/down/left/right), the default.
            - ``"8"`` — king's moves (4-connected plus diagonals).
            - ``"triangular"`` — triangular lattice matching Binney et al.
              (2013) Fig. 3.  Odd rows are shifted by half a lon-step and
              each node connects to its 6 nearest neighbours, producing
              approximately equilateral triangles.

            In 3-D mode (``n_depth > 1``), each depth layer uses the
            requested horizontal connectivity, and adjacent layers at
            the same (lat, lon) position are connected vertically.
        seed:
            Unused for grids but accepted for a uniform interface.
        """
        _ = seed  # grids are deterministic without RNG

        lat_min = float(region["lat_min"])
        lat_max = float(region["lat_max"])
        lon_min = float(region["lon_min"])
        lon_max = float(region["lon_max"])

        n_lat = int(n_lat)
        n_lon = int(n_lon)
        n_depth = int(n_depth)

        lats = np.linspace(lat_min, lat_max, n_lat)
        lons = np.linspace(lon_min, lon_max, n_lon)
        lon_step = lons[1] - lons[0] if n_lon > 1 else 0.0

        use_depth = n_depth > 1
        if use_depth:
            depth_min = float(region.get("depth_min", 0.0))
            depth_max = float(region.get("depth_max", 100.0))
            depths = np.linspace(depth_min, depth_max, n_depth)
        else:
            depths = np.array([None])

        n_horiz = n_lat * n_lon  # nodes per depth layer
        g = nx.Graph()

        # Add nodes.  For "triangular", odd rows are shifted by half a
        # lon-step to form equilateral triangles.
        triangular = connectivity == "triangular"
        for k, dep in enumerate(depths):
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    node_id = k * n_horiz + i * n_lon + j
                    if triangular and i % 2 == 1:
                        lon_shifted = float(lon) + 0.5 * lon_step
                    else:
                        lon_shifted = float(lon)
                    attrs = {"lat": float(lat), "lon": lon_shifted}
                    if use_depth:
                        attrs["depth"] = float(dep)
                    g.add_node(node_id, **attrs)

        # Add edges depending on connectivity.
        for k in range(n_depth if use_depth else 1):
            base = k * n_horiz
            for i in range(n_lat):
                for j in range(n_lon):
                    node_id = base + i * n_lon + j

                    # Horizontal neighbour (shared by all modes).
                    if j + 1 < n_lon:
                        cls._add_edge_with_distance(g, node_id, base + i * n_lon + (j + 1), speed_mps)

                    # Vertical neighbour (shared by all modes).
                    if i + 1 < n_lat:
                        cls._add_edge_with_distance(g, node_id, base + (i + 1) * n_lon + j, speed_mps)

                    if connectivity == "8":
                        # Diagonal neighbours.
                        if i + 1 < n_lat and j + 1 < n_lon:
                            cls._add_edge_with_distance(g, node_id, base + (i + 1) * n_lon + (j + 1), speed_mps)
                        if i + 1 < n_lat and j - 1 >= 0:
                            cls._add_edge_with_distance(g, node_id, base + (i + 1) * n_lon + (j - 1), speed_mps)

                    elif triangular:
                        if i + 1 < n_lat:
                            if i % 2 == 0:
                                if j + 1 < n_lon:
                                    cls._add_edge_with_distance(
                                        g, node_id, base + (i + 1) * n_lon + (j + 1), speed_mps
                                    )
                            else:
                                if j - 1 >= 0:
                                    cls._add_edge_with_distance(
                                        g, node_id, base + (i + 1) * n_lon + (j - 1), speed_mps
                                    )

            # Vertical (depth) edges connecting this layer to the next one.
            if use_depth and k + 1 < n_depth:
                next_base = (k + 1) * n_horiz
                for i in range(n_lat):
                    for j in range(n_lon):
                        cls._add_edge_with_distance(
                            g,
                            base + i * n_lon + j,
                            next_base + i * n_lon + j,
                            speed_mps,
                        )

        return cls(graph=g, speed_mps=float(speed_mps))

    @classmethod
    def random_geometric(
        cls,
        region: Mapping[str, float],
        n_nodes: int,
        *,
        k: Optional[int] = None,
        radius: Optional[float] = None,
        speed_mps: float,
        seed: Optional[int] = None,
    ) -> "WaypointGraph":
        """
        Build a random geometric graph over the region.

        Nodes are sampled uniformly in the bounding box. Edges connect either:

        - the ``k`` nearest neighbours for each node (if ``k`` is provided), or
        - all nodes within a given great-circle distance ``radius`` (meters).

        If the *region* contains ``depth_min`` and ``depth_max``, nodes are
        sampled uniformly in 3-D and the ``depth`` attribute is stored.
        """
        if k is None and radius is None:
            raise ValueError("Either k or radius must be provided for random_geometric.")

        rng = np.random.default_rng(seed)

        lat_min = float(region["lat_min"])
        lat_max = float(region["lat_max"])
        lon_min = float(region["lon_min"])
        lon_max = float(region["lon_max"])

        use_depth = "depth_min" in region and "depth_max" in region

        lats = rng.uniform(lat_min, lat_max, int(n_nodes))
        lons = rng.uniform(lon_min, lon_max, int(n_nodes))

        g = nx.Graph()
        if use_depth:
            depth_min = float(region["depth_min"])
            depth_max = float(region["depth_max"])
            node_depths = rng.uniform(depth_min, depth_max, int(n_nodes))
            for node_id, (lat, lon, dep) in enumerate(zip(lats, lons, node_depths)):
                g.add_node(node_id, lat=float(lat), lon=float(lon), depth=float(dep))
            coords = np.column_stack([lats, lons, node_depths])
        else:
            for node_id, (lat, lon) in enumerate(zip(lats, lons)):
                g.add_node(node_id, lat=float(lat), lon=float(lon))
            coords = np.column_stack([lats, lons])

        if k is not None:
            # Connect to k nearest neighbours in Euclidean lat-lon(-depth) space
            # (sufficient for small regions).
            from sklearn.neighbors import NearestNeighbors  # optional dependency

            nn = NearestNeighbors(n_neighbors=int(k) + 1, metric="euclidean")
            nn.fit(coords)
            distances, indices = nn.kneighbors(coords, return_distance=True)
            n = coords.shape[0]
            for i in range(n):
                for j_idx in range(1, indices.shape[1]):  # skip self at j_idx=0
                    j = int(indices[i, j_idx])
                    if i == j:
                        continue
                    cls._add_edge_with_distance(g, i, j, speed_mps)
        else:
            # Radius-based connections using great-circle (+ depth) distance.
            rad_m = float(radius)
            n = coords.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    d_args: dict = dict(
                        lat1_deg=float(coords[i, 0]),
                        lon1_deg=float(coords[i, 1]),
                        lat2_deg=float(coords[j, 0]),
                        lon2_deg=float(coords[j, 1]),
                    )
                    if use_depth:
                        d_args["depth1_m"] = float(coords[i, 2])
                        d_args["depth2_m"] = float(coords[j, 2])
                    d = cls._haversine_m(**d_args)
                    if d <= rad_m:
                        cls._add_edge(g, i, j, d, speed_mps)

        return cls(graph=g, speed_mps=float(speed_mps))

    # ------------------------------------------------------------------
    # Core graph utilities
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def nodes(self) -> Iterable[int]:
        return self.graph.nodes

    def node_coords(self, node: int) -> Tuple[float, float]:
        data = self.graph.nodes[node]
        return float(data["lat"]), float(data["lon"])

    def node_depth(self, node: int) -> Optional[float]:
        """Return the depth of *node* in meters, or ``None`` if not set."""
        d = self.graph.nodes[node].get("depth")
        return float(d) if d is not None else None

    @property
    def has_depth(self) -> bool:
        """``True`` if any node carries a ``depth`` attribute."""
        return any("depth" in d for _, d in self.graph.nodes(data=True))

    def edge_attributes(self, u: int, v: int) -> Mapping[str, float]:
        return self.graph.edges[u, v]

    def shortest_time(self, s: int, t: int) -> float:
        """
        Shortest-path travel time between nodes s and t (seconds).
        """
        return float(
            nx.shortest_path_length(
                self.graph, s, t, weight="time_s"  # type: ignore[arg-type]
            )
        )

    def shortest_path(self, s: int, t: int) -> List[int]:
        """
        Shortest-time path between nodes s and t as a list of node ids.
        """
        return list(nx.shortest_path(self.graph, s, t, weight="time_s"))

    def path_travel_time(self, path: Sequence[int]) -> float:
        """
        Total travel time along a path given as a sequence of node ids.
        """
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            data = self.graph.edges[u, v]
            total += float(data["time_s"])
        return total

    def is_feasible(self, s: int, t: int, budget_seconds: float) -> bool:
        """
        Return True if the shortest-path travel time from s to t is within B.
        """
        try:
            d = self.shortest_time(s, t)
        except nx.NetworkXNoPath:
            return False
        return d <= float(budget_seconds)

    def serialize_graph(self) -> Mapping[str, object]:
        """
        Serialize graph metadata to a JSON-serializable dictionary for logging.
        """
        nodes = []
        for nid, data in self.graph.nodes(data=True):
            entry: dict = {
                "id": int(nid),
                "lat": float(data["lat"]),
                "lon": float(data["lon"]),
            }
            if "depth" in data:
                entry["depth"] = float(data["depth"])
            nodes.append(entry)

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append(
                {
                    "u": int(u),
                    "v": int(v),
                    "length_m": float(data["length_m"]),
                    "time_s": float(data["time_s"]),
                }
            )

        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "speed_mps": float(self.speed_mps),
            "nodes": nodes,
            "edges": edges,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine_m(
        lat1_deg: float,
        lon1_deg: float,
        lat2_deg: float,
        lon2_deg: float,
        depth1_m: Optional[float] = None,
        depth2_m: Optional[float] = None,
    ) -> float:
        """
        Distance in meters, optionally including a depth component.

        The horizontal distance uses the haversine (great-circle) formula on a
        spherical Earth.  When both *depth1_m* and *depth2_m* are provided the
        result is ``sqrt(haversine**2 + (depth2 - depth1)**2)``.
        """
        r_earth = 6371e3  # meters
        lat1 = np.deg2rad(lat1_deg)
        lon1 = np.deg2rad(lon1_deg)
        lat2 = np.deg2rad(lat2_deg)
        lon2 = np.deg2rad(lon2_deg)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        horiz = float(r_earth * c)

        if depth1_m is not None and depth2_m is not None:
            dz = float(depth2_m) - float(depth1_m)
            return float(np.sqrt(horiz**2 + dz**2))
        return horiz

    @classmethod
    def _add_edge_with_distance(
        cls,
        g: nx.Graph,
        u: int,
        v: int,
        speed_mps: float,
    ) -> None:
        nu, nv = g.nodes[u], g.nodes[v]
        lat_u, lon_u = float(nu["lat"]), float(nu["lon"])
        lat_v, lon_v = float(nv["lat"]), float(nv["lon"])
        dep_u = nu.get("depth")
        dep_v = nv.get("depth")
        depth1 = float(dep_u) if dep_u is not None else None
        depth2 = float(dep_v) if dep_v is not None else None
        length_m = cls._haversine_m(lat_u, lon_u, lat_v, lon_v, depth1, depth2)
        cls._add_edge(g, u, v, length_m, speed_mps)

    @staticmethod
    def _add_edge(
        g: nx.Graph,
        u: int,
        v: int,
        length_m: float,
        speed_mps: float,
    ) -> None:
        time_s = float(length_m) / float(speed_mps) if speed_mps > 0 else np.inf
        g.add_edge(u, v, length_m=float(length_m), time_s=float(time_s))

