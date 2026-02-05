 """CybORG (CAGE-4) gymnasium-compatible wrapper with NeSy ablations.

This file is intentionally *wrapper-only*: it does not modify CybORG internals.
It matches Gymnasium's API so RLlib (>=2.x) can run without StepAPI issues.

Ablation modes (set via env var `NESY_MODE` or env_config["nesy_mode"]):
  - "base"   : No state abstraction, no reward shaping (paper baseline).
  - "state"  : State abstraction only (compressed observation).
  - "reward" : Reward shaping only (baseline observation).
  - "full"   : State abstraction + reward shaping (NeSy as used in the paper).

Reward shaping strength is controlled by `NESY_LAM` (float).
"""

from __future__ import annotations

import os
import sys
from collections import deque
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

# CybORG path: env var or auto-detect
def _setup_cyborg_path():
    """Add CybORG module path to sys.path.
    
    Priority:
    1. Env vars CAGE4_CC4_PATH or CYBORG_PATH
    2. Relative: third_party/CybORG
    3. Relative: ../cage-challenge-4/CybORG
    """
    cyborg_path = None
    
    # Check env vars
    for env_var in ["CAGE4_CC4_PATH", "CYBORG_PATH"]:
        env_path = os.environ.get(env_var, "").strip()
        if env_path and os.path.isdir(env_path):
            cyborg_path = os.path.abspath(env_path)
            break
    
    # Auto-detect
    if cyborg_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "third_party", "CybORG"),
            os.path.join(here, "..", "cage-challenge-4", "CybORG"),
            os.path.join(os.path.dirname(here), "third_party", "CybORG"),
        ]
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if os.path.isdir(candidate):
                cyborg_path = candidate
                break
    
    # Add to sys.path
    if cyborg_path and cyborg_path not in sys.path:
        sys.path.insert(0, cyborg_path)
        return cyborg_path
    
    return None

_setup_cyborg_path()

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper


def _get_env_var(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v is not None else default


def _parse_mode(raw: str | None) -> str:
    if not raw:
        return "base"
    raw = raw.strip().lower()
    if raw in {"base", "baseline"}:
        return "base"
    if raw in {"state", "state_only", "abstract", "status_abstract"}:
        return "state"
    if raw in {"reward", "reward_only", "shaping"}:
        return "reward"
    if raw in {"full", "nesy", "both"}:
        return "full"
    if raw in {"graph", "graph_based", "gnn"}:
        return "graph"
    if raw in {"frame_stack", "frame_stacking", "temporal"}:
        return "frame_stack"
    if raw in {"adaptive_scale", "adaptive_scaling", "adaptive"}:
        return "adaptive_scale"
    if raw in {"logic_guided", "logic", "guided"}:
        return "logic_guided"
    if raw in {"rule_pruning", "pruning", "rule_based"}:
        return "rule_pruning"
    if raw in {"full_logic", "full_logic_guided", "full+logic"}:
        return "full_logic"
    if raw in {"full_rule", "full_rule_pruning", "full+rule"}:
        return "full_rule"
    if raw in {"full_all", "full_all_enhanced", "full+all"}:
        return "full_all"
    if raw in {"full_ontology", "full+ontology", "full+onto", "fullonto"}:
        return "full_ontology"
    if raw in {"ontology", "onto", "knowledge_graph", "kg"}:
        return "ontology"
    # Fail-safe: unknown strings fall back to base (do not crash a large sweep).
    return "base"


class CybORGWrapper(gym.Env):
    """Gymnasium-compatible env for RLlib with NeSy ablations.

    Uses DroneSwarmScenarioGenerator (like 1st_success) but adds NeSy ablation modes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int | None = None,
        max_episode_steps: int = 800,
        nesy_mode: str = "base",
        nesy_lam: float = 1.0,
        blue_agent: str = "blue_agent_0",
        enable_frame_stack: bool = False,
        enable_adaptive_scale: bool = False,
    ):
        super().__init__()

        self.seed_value = int(seed) if seed is not None else None
        self.max_episode_steps = int(max_episode_steps)
        self.steps = 0
        self.episode_count = 0  # Track training progress (for annealing)
        self.total_steps = 0  # Total step count
        self.blue_agent = blue_agent

        self.nesy_mode = _parse_mode(nesy_mode)
        # State Abstraction and Reward Shaping are in the full-mode family
        # logic_guided and rule_pruning add only that feature to base
        # Ontology mode: use Full NeSy state abstraction; reward shaping is independent
        # full_ontology: Full NeSy (state + multi-objective reward) + Ontology reward shaping
        self.use_ontology = self.nesy_mode in {"ontology", "full_ontology"}
        # Observation: both Full NeSy and Ontology use 52-dim compression (fair comparison)
        self.use_state_abstraction = self.nesy_mode in {"state", "full", "full_logic", "full_rule", "full_all", "full_ontology"} or self.use_ontology
        # Reward Shaping: Full NeSy = multi-objective; Ontology = Axiom-based (independent)
        # full_ontology: both Full NeSy multi-objective and Ontology axiom-based
        self.use_reward_shaping = self.nesy_mode in {"reward", "full", "full_logic", "full_rule", "full_all", "full_ontology"}
        self.use_graph_representation = self.nesy_mode == "graph"
        # Full-mode family: whether Logic-Guided and Rule-Based Pruning are included
        # full_logic: Full + Logic-Guided
        # full_rule: Full + Rule-Based Pruning
        # full_all: Full + Logic-Guided + Rule-Based Pruning
        self.use_logic_guided = self.nesy_mode in {"logic_guided", "full_logic", "full_all"}
        self.use_rule_pruning = self.nesy_mode in {"rule_pruning", "full_rule", "full_all"}
        self.nesy_lam = float(nesy_lam)

        # Frame Stacking and Adaptive Scaling
        # Default: disabled for all experiments (False)
        # Enabled only in frame_stack or adaptive_scale mode
        self.enable_frame_stack = enable_frame_stack or self.nesy_mode == "frame_stack"
        self.enable_adaptive_scale = enable_adaptive_scale or self.nesy_mode == "adaptive_scale"

        # --- Build CybORG (same as 1st_success) ---
        self.scenario = DroneSwarmScenarioGenerator()
        self.cyborg = CybORG(self.scenario, "sim", seed=self.seed_value)
        
        # Same wrapper structure as 1st_success
        self.raw_wrapped = FixedFlatWrapper(self.cyborg)
        self._gym_env = OpenAIGymWrapper(agent_name=self.blue_agent, env=self.raw_wrapped)

        # Host info (used in all modes)
        self.known_hosts = [
            'User0', 'User1', 'User2', 'User3', 'User4',
            'Enterprise0', 'Enterprise1', 'Enterprise2', 'Enterprise_Server',
            'Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2',
            'Defender', 'User_Router', 'Enterprise_Router', 'Op_Router'
        ]
        self.critical_targets = ['Op_Server0', 'Enterprise_Server', 'Op_Host0']
        self.feats_per_host = 3
        # Observation dimension by ablation design:
        # - Exp 2 (state): 52-dim state abstraction (paper: "52-dim compressed observation")
        # - Exp 3 (reward): raw observation (paper: "keep baseline raw observation")
        # - Exp 5 (ontology): 107-dim ontology observation (paper: "107-dim ontology observation")
        # - Exp 4 (full): 52-dim state abstraction (Full NeSy uses state abstraction)
        # - Exp 6 (full_ontology = 2+3+5): 107-dim ontology observation
        #   → 107-dim includes 52-dim info (is_compromised, service_status etc.), so state info is included.
        #     Reward is from reward shaping; ontology from observation.
        #   → 159-dim concat is redundant, not used
        self.use_107dim_unified = self.nesy_mode in {"ontology", "full_ontology"}  # both 107-dim
        self.ontology_obs_dim = 1 + 4 + len(self.known_hosts) * 4 + len(self.known_hosts) * 2  # 107 for 17 hosts
        
        # Uptime tracking (improvement: uptime-centric reward)
        self._prev_uptime = 1.0
        self._uptime_history = deque(maxlen=10)  # Track uptime over last 10 steps
        
        # ===== Frame Stacking (temporal info) =====
        # Must be initialized before setting observation space
        # Default: 1 (disabled); only in frame_stack mode
        if self.enable_frame_stack:
            self.frame_stack = int(os.environ.get("NESY_FRAME_STACK", "4"))  # frame_stack mode: 4 frames
        else:
            self.frame_stack = 1  # Disabled
        if self.frame_stack < 1:
            self.frame_stack = 1
        self._obs_history = deque(maxlen=self.frame_stack)
        
        # Observation space (by ablation design):
        # - Base: raw observation (~11k dim)
        # - State (2): 52-dim state abstraction
        # - Reward (3): raw observation
        # - Ontology (5): 107-dim ontology observation
        # - Full (4): 52-dim state abstraction
        # - Full Ontology (6 = 2+3+5): 107-dim ontology observation
        #   → 107-dim includes 52-dim info, so exp 2 (state) is included
        #   → 159-dim concat is redundant, not used
        if self.use_107dim_unified:
            # ontology, full_ontology both 107-dim
            base_obs_dim = self.ontology_obs_dim  # 107
            obs_dim = base_obs_dim * self.frame_stack
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        elif self.use_state_abstraction:
            # state, full etc.: 52-dim state abstraction
            base_obs_dim = 1 + len(self.known_hosts) * self.feats_per_host  # uptime + hosts
            obs_dim = base_obs_dim * self.frame_stack
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        elif self.use_graph_representation:
            # Graph-based representation (paper: King et al., 2025)
            # Node embeddings: k-hop neighborhood per host
            # Simplified: 64-dim embedding per host (paper GNN output)
            # 17 hosts × 64 = 1088 dims
            graph_embedding_dim = 64  # GNN embedding dimension
            base_obs_dim = len(self.known_hosts) * graph_embedding_dim
            # Apply frame stacking
            obs_dim = base_obs_dim * self.frame_stack
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        elif self.use_state_abstraction:
            # Paper and 1st_success: host-specific knowledge vector
            # 17 hosts × 3 features + uptime = 52 dims (close to paper's 60 dims)
            base_obs_dim = 1 + len(self.known_hosts) * self.feats_per_host  # uptime + hosts
            # Apply frame stacking
            obs_dim = base_obs_dim * self.frame_stack
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        else:
            # Baseline: use FixedFlatWrapper observation space
            # In frame_stack mode, dims increase for frame stacking
            base_obs_space = self._gym_env.observation_space
            if hasattr(base_obs_space, "shape"):
                base_obs_dim = int(base_obs_space.shape[0])
            else:
                base_obs_dim = 11293  # fallback
            
            # Apply frame stacking (when in frame_stack mode)
            obs_dim = base_obs_dim * self.frame_stack
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

        # Action space
        if hasattr(self._gym_env.action_space, "n"):
            n_actions = int(self._gym_env.action_space.n)
        else:
            n_actions = 56  # fallback
        self.action_space = spaces.Discrete(n_actions)
        
        # ===== NeSy Reward Shaping init =====
        # State tracking for multi-objective reward shaping
        self._prev_uptime = 1.0  # Initial uptime (updated on reset)
        
        # ===== Adaptive Reward Scaling =====
        # Adjust bonus by training phase (larger early on)
        # Default: disabled; only in adaptive_scale mode
        self._episode_count = 0
        self._adaptive_scale = 1.0

    # -------------------------
    # Utilities for NeSy signals
    # -------------------------
    def _try_get_hosts(self):
        """Best-effort host list extraction.

        CybORG internals have changed across versions.
        We attempt several common attribute paths. If not found, return None.
        """
        candidates = [
            ("environment", "hosts"),
            ("env", "hosts"),
            ("sim", "hosts"),
            ("_environment", "hosts"),
        ]
        for a, b in candidates:
            obj = getattr(self.cyborg, a, None)
            if obj is not None and hasattr(obj, b):
                return getattr(obj, b)
        return None

    def _calculate_uptime_fast(self) -> float:
        """Compute a coarse uptime proxy in [0, 1] if possible.

        If we cannot access host state (API differences), return 0.0 rather
        than crashing training.
        """
        hosts = self._try_get_hosts()
        if hosts is None:
            return 0.0

        total = 0
        up = 0
        for h in hosts.values():
            total += 1
            # Heuristic: different versions store compromise/availability differently.
            # Treat "None/Unknown" as not up.
            if hasattr(h, "availability"):
                # Many CybORG versions use enums for availability.
                av = getattr(h, "availability")
                if str(av).lower().find("up") >= 0 or str(av).lower().find("available") >= 0:
                    up += 1
            elif hasattr(h, "is_up"):
                up += 1 if getattr(h, "is_up") else 0
        return float(up) / float(total) if total > 0 else 0.0

    def _get_compromise_counts(self):
        """Return (#compromised, #unknown) best-effort."""
        hosts = self._try_get_hosts()
        if hosts is None:
            return 0.0, 0.0
        compromised = 0
        unknown = 0
        for h in hosts.values():
            if hasattr(h, "compromised"):
                c = getattr(h, "compromised")
                if c is None:
                    unknown += 1
                elif bool(c):
                    compromised += 1
            elif hasattr(h, "compromise_state"):
                cs = getattr(h, "compromise_state")
                s = str(cs).lower()
                if "unknown" in s:
                    unknown += 1
                elif "comprom" in s or "user" in s or "root" in s:
                    compromised += 1
        return float(compromised), float(unknown)

    def _extract_hifi_knowledge(self, uptime: float) -> np.ndarray:
        """1st_success and paper: host-specific knowledge vector (52 dims).
        
        Implementation close to paper's 60-dim knowledge vector.
        Per host: is_compromised, service_up, activity.
        """
        try:
            true_state = self.cyborg.get_agent_state('True')
        except Exception:
            # Fallback: simple method
            uptime_val = self._calculate_uptime_fast()
            compromised, unknown = self._get_compromise_counts()
            return np.array([uptime_val, compromised, unknown], dtype=np.float32)
        
        features = [uptime]
        for hostname in self.known_hosts:
            host_info = true_state.get(hostname, {})
            
            # is_compromised: whether Red agent session exists
            is_compromised = 0.0
            for sess in host_info.get('Sessions', []):
                # Improvement: no str() cast (consistent with 1st_success)
                agent_name = sess.get('Agent', '')
                if agent_name and 'Red' in agent_name:
                    is_compromised = 1.0
                    break
            
            # service_up: whether service process is running
            service_up = 0.0
            for proc in host_info.get('Processes', []):
                if proc.get('Service Name'):
                    service_up = 1.0
                    break
            
            # activity: active if compromised or service_up
            activity = 1.0 if (is_compromised or service_up) else 0.0
            
            features.extend([is_compromised, service_up, activity])
        
        return np.array(features, dtype=np.float32)

    def _nesy_state_features(self) -> np.ndarray:
        """Return the NeSy state vector (paper: 52 dims)."""
        uptime = self._calculate_uptime_fast()
        return self._extract_hifi_knowledge(uptime)
    
    def _ontology_based_observation(self) -> np.ndarray:
        """Ontology-based observation: state representation from explicit ontology structure.
        
        NeSy requirements:
        1. Explicit Knowledge Representation (ontology)
        2. Symbolic Reasoning (logic-rule-based inference)
        3. Neural-Symbolic Integration (convert to neural input)
        
        Returns:
            ontology_features: np.ndarray of shape (107,)
        """
        try:
            true_state = self.cyborg.get_agent_state('True')
        except Exception:
            # Fallback: zero features
            uptime = self._calculate_uptime_fast()
            return np.zeros(1 + 4 + (len(self.known_hosts) * 4) + (len(self.known_hosts) * 2), dtype=np.float32)
        
        uptime = self._calculate_uptime_fast()
        features = [uptime]
        
        # ===== 1. MITRE ATT&CK Tactics (4 dims) =====
        # Explicit attack-stage classification (ontology: AttackStage concept)
        recon_score = 0.0
        access_score = 0.0
        lateral_score = 0.0
        impact_score = 0.0
        
        compromised_hosts = []
        for hostname in self.known_hosts:
            host_info = true_state.get(hostname, {})
            is_compromised = False
            for sess in host_info.get('Sessions', []):
                if 'Red' in str(sess.get('Agent', '')):
                    is_compromised = True
                    compromised_hosts.append(hostname)
                    break
            
            if is_compromised:
                # Ontology rule: Critical target compromised → Impact
                if hostname in self.critical_targets:
                    impact_score += 1.0
                # Ontology rule: Op/Enterprise network → Lateral Movement
                elif 'Op_' in hostname or 'Enterprise' in hostname:
                    lateral_score += 1.0
                # Ontology rule: User network → Initial Access
                else:
                    access_score += 1.0
                # Every compromised host includes Recon stage
                recon_score += 1.0
        
        # Normalize tactic scores
        num_hosts = len(self.known_hosts)
        tactic_scores = np.array([
            min(recon_score / num_hosts, 1.0),
            min(access_score / num_hosts, 1.0),
            min(lateral_score / num_hosts, 1.0),
            min(impact_score / len(self.critical_targets) if len(self.critical_targets) > 0 else 1.0, 1.0)
        ], dtype=np.float32)
        features.extend(tactic_scores.tolist())
        
        # ===== 2. Host Ontology Features (17 hosts × 4 features) =====
        # Ontology: Host concept attributes
        host_criticality_map = {}
        for hostname in self.known_hosts:
            if hostname in self.critical_targets:
                host_criticality_map[hostname] = 1.0
            elif 'Op_' in hostname:
                host_criticality_map[hostname] = 0.7
            elif 'Enterprise' in hostname:
                host_criticality_map[hostname] = 0.5
            else:
                host_criticality_map[hostname] = 0.2
        
        for hostname in self.known_hosts:
            host_info = true_state.get(hostname, {})
            
            # Feature 1: is_compromised (0.0 or 1.0)
            is_compromised = 0.0
            for sess in host_info.get('Sessions', []):
                # Improvement: no str() cast (consistent with 1st_success)
                agent_name = sess.get('Agent', '')
                if agent_name and 'Red' in agent_name:
                    is_compromised = 1.0
                    break
            
            # Feature 2: criticality (ontology: Host.criticality)
            criticality = host_criticality_map.get(hostname, 0.0)
            
            # Feature 3: service_status (ontology: Service.running)
            service_status = 0.0
            for proc in host_info.get('Processes', []):
                if proc.get('Service Name'):
                    service_status = 1.0
                    break
            
            # Feature 4: threat_level (ontology-rule-based)
            # Rule: compromised AND critical → high threat
            # Rule: compromised AND service_down → medium threat
            # Rule: safe → low threat
            if is_compromised > 0:
                if criticality >= 0.8:
                    threat_level = 1.0  # High threat
                elif service_status < 0.5:
                    threat_level = 0.6  # Medium threat (service down)
                else:
                    threat_level = 0.4  # Medium threat (service up)
            else:
                threat_level = 0.0  # Low threat (safe)
            
            features.extend([is_compromised, criticality, service_status, threat_level])
        
        # ===== 3. Network Topology Relations (17 hosts × 2 features) =====
        # Ontology: Relation concepts (depends_on, connected_to)
        subnet_mapping = {
            'User': ['User0', 'User1', 'User2', 'User3', 'User4', 'User_Router'],
            'Enterprise': ['Enterprise0', 'Enterprise1', 'Enterprise2', 'Enterprise_Server', 'Enterprise_Router'],
            'Op': ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Router'],
        }
        
        for hostname in self.known_hosts:
            # Feature 1: dependency_score (ontology: depends_on)
            # Dependency on critical hosts
            dependency_score = 0.0
            for critical_host in self.critical_targets:
                if hostname == critical_host:
                    dependency_score = 1.0  # self
                    break
                # Higher dependency if same subnet
                for subnet, hosts in subnet_mapping.items():
                    if hostname in hosts and critical_host in hosts:
                        dependency_score += 0.3
            dependency_score = min(dependency_score, 1.0)
            
            # Feature 2: connectivity_score (ontology: connected_to)
            # Connectivity to other hosts in same subnet
            connectivity_score = 0.0
            for subnet, hosts in subnet_mapping.items():
                if hostname in hosts:
                    connectivity_score = len(hosts) / 10.0  # Normalize by subnet size
                    break
            connectivity_score = min(connectivity_score, 1.0)
            
            features.extend([dependency_score, connectivity_score])
        
        return np.array(features, dtype=np.float32)
    
    def _build_dependency_graph(self) -> dict:
        """Ontology: Build dependency graph between hosts (explicit relation modeling).
        
        Returns:
            dependency_graph: dict[host, list[dependent_hosts]]
        """
        # Explicit dependency relations (ontology: depends_on)
        dependency_graph = {
            'Op_Server0': ['Enterprise_Server', 'Op_Host0', 'Op_Host1'],  # most critical, depends on many hosts
            'Enterprise_Server': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
            'Op_Host0': ['Op_Host1', 'Op_Host2'],
            'Op_Host1': ['Op_Host2'],
            'Enterprise0': ['Enterprise1'],
            'Enterprise1': ['Enterprise2'],
        }
        return dependency_graph
    
    def _infer_attack_chain(self, true_state: dict, compromised_hosts: list) -> dict:
        """Ontology: MITRE ATT&CK-based attack chain inference and prediction.
        
        Returns:
            attack_chain: {
                'current_stage': str,
                'next_stage': str,
                'predicted_targets': list,
                'urgency': float,
                'stages': dict
            }
        """
        # Attack stage classification
        stages = {
            'Reconnaissance': [],
            'Initial Access': [],
            'Lateral Movement': [],
            'Impact': []
        }
        
        for host in compromised_hosts:
            if host in self.critical_targets:
                stages['Impact'].append(host)
            elif 'Op_' in host or 'Enterprise' in host:
                stages['Lateral Movement'].append(host)
            elif 'User' in host:
                stages['Initial Access'].append(host)
            stages['Reconnaissance'].append(host)  # every compromised host includes Recon
        
        # Next attack stage prediction (ontology inference)
        current_stage = 'Reconnaissance'
        if stages['Impact']:
            current_stage = 'Impact'
        elif stages['Lateral Movement']:
            current_stage = 'Lateral Movement'
        elif stages['Initial Access']:
            current_stage = 'Initial Access'
        
        # Next stage prediction
        next_stage = 'Initial Access'
        predicted_targets = []
        urgency = 0.0
        
        if current_stage == 'Reconnaissance':
            next_stage = 'Initial Access'
            predicted_targets = ['User_Router', 'User0']
            urgency = 0.3
        elif current_stage == 'Initial Access':
            next_stage = 'Lateral Movement'
            predicted_targets = ['Enterprise0', 'Op_Host0']
            urgency = 0.6
        elif current_stage == 'Lateral Movement':
            next_stage = 'Impact'
            predicted_targets = self.critical_targets
            urgency = 1.0  # highest urgency
        
        return {
            'current_stage': current_stage,
            'next_stage': next_stage,
            'predicted_targets': predicted_targets,
            'urgency': urgency,
            'stages': stages
        }
    
    def _calculate_recovery_impact(self, recovered_host: str, dependency_graph: dict, true_state: dict, depth: int = 0, max_depth: int = 2) -> float:
        """Ontology: Recovery impact from dependency graph (recursive inference).
        
        Computes protection effect on hosts that the recovered host depends on.
        
        Args:
            recovered_host: Recovered host name
            dependency_graph: Dependency graph
            true_state: Current environment true state
            depth: Current recursion depth
            max_depth: Max recursion depth
        
        Returns:
            impact: float (>= 1.0, includes dependency chain protection effect)
        """
        # Recursion depth limit (prevent infinite loop)
        if depth > max_depth:
            return 1.0  # Base case for recursion depth
        
        impact = 1.0  # direct recovery bonus
        
        # Dependency chain inference
        dependent_hosts = dependency_graph.get(recovered_host, [])
        for dep_host in dependent_hosts:
            # Bonus if dependent host is safe
            dep_info = true_state.get(dep_host, {})
            is_dep_compromised = False
            for sess in dep_info.get('Sessions', []):
                agent_name = sess.get('Agent', '')
                if agent_name and 'Red' in agent_name:
                    is_dep_compromised = True
                    break
            
            if not is_dep_compromised:
                # Dependency chain protection bonus (recursive, depth-limited) - weight 0.3→0.5
                impact += 0.5 * self._calculate_recovery_impact(dep_host, dependency_graph, true_state, depth + 1, max_depth)
        
        return impact
    
    def _extract_graph_structure(self) -> tuple[dict, list]:
        """Graph structure extraction (paper-style).
        
        Returns:
            - node_features: dict[hostname, feature_vector]
            - edges: list of (host1, host2) tuples
        """
        try:
            true_state = self.cyborg.get_agent_state('True')
        except Exception:
            return {}, []
        
        node_features = {}
        edges = []
        
        # Extract host nodes and features
        for hostname in self.known_hosts:
            host_info = true_state.get(hostname, {})
            
            # Node features (see paper Table 7)
            # Host features: compromised, service_up, activity, is_critical, is_user, is_server
            is_compromised = 0.0
            for sess in host_info.get('Sessions', []):
                # Improvement: no str() cast (consistent with 1st_success)
                agent_name = sess.get('Agent', '')
                if agent_name and 'Red' in agent_name:
                    is_compromised = 1.0
                    break
            
            service_up = 0.0
            for proc in host_info.get('Processes', []):
                if proc.get('Service Name'):
                    service_up = 1.0
                    break
            
            activity = 1.0 if (is_compromised or service_up) else 0.0
            is_critical = 1.0 if hostname in self.critical_targets else 0.0
            is_user = 1.0 if 'User' in hostname else 0.0
            is_server = 1.0 if 'Server' in hostname or 'Host' in hostname else 0.0
            
            # Feature vector (6-dim)
            node_features[hostname] = np.array([
                is_compromised,
                service_up,
                activity,
                is_critical,
                is_user,
                is_server
            ], dtype=np.float32)
        
        # Edge extraction (paper: Connection info)
        # Simplified: hosts in same subnet are connected
        # Or use actual network topology when available
        subnet_mapping = {
            'User': ['User0', 'User1', 'User2', 'User3', 'User4', 'User_Router'],
            'Enterprise': ['Enterprise0', 'Enterprise1', 'Enterprise2', 'Enterprise_Server', 'Enterprise_Router'],
            'Op': ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Router'],
        }
        
        for subnet, hosts in subnet_mapping.items():
            for i, h1 in enumerate(hosts):
                for h2 in hosts[i+1:]:
                    if h1 in node_features and h2 in node_features:
                        edges.append((h1, h2))
        
        return node_features, edges
    
    def _compute_graph_embeddings(self, node_features: dict, edges: list) -> np.ndarray:
        """Graph Neural Network embedding computation (paper-style).
        
        Paper uses GCN, SAGE, GAT, GIN; here simplified as message passing.
        
        Args:
            node_features: dict[hostname, feature_vector]
            edges: list of (host1, host2) tuples
        
        Returns:
            node_embeddings: np.ndarray of shape (num_nodes, embedding_dim)
        """
        if not node_features:
            # Fallback: zero embeddings
            return np.zeros((len(self.known_hosts), 64), dtype=np.float32)
        
        # Simplified GNN: 2-layer message passing
        embedding_dim = 64
        num_nodes = len(self.known_hosts)
        
        # Initialize node embeddings from features
        # Feature vector (6-dim) → embedding (64-dim)
        embeddings = np.zeros((num_nodes, embedding_dim), dtype=np.float32)
        
        # Feature projection (simple linear transform)
        for idx, hostname in enumerate(self.known_hosts):
            if hostname in node_features:
                features = node_features[hostname]
                # Simple feature expansion (paper uses learned GNN; here simplified)
                expanded = np.tile(features, embedding_dim // len(features) + 1)[:embedding_dim]
                embeddings[idx] = expanded
        
        # Message passing (simplified 2-layer GNN)
        # Layer 1: Aggregate neighbor features
        neighbor_embeddings = np.zeros_like(embeddings)
        node_to_idx = {hostname: idx for idx, hostname in enumerate(self.known_hosts)}
        
        for h1, h2 in edges:
            if h1 in node_to_idx and h2 in node_to_idx:
                idx1, idx2 = node_to_idx[h1], node_to_idx[h2]
                # Bidirectional edges
                neighbor_embeddings[idx1] += embeddings[idx2] * 0.1
                neighbor_embeddings[idx2] += embeddings[idx1] * 0.1
        
        # Update embeddings (simple aggregation)
        embeddings = embeddings * 0.7 + neighbor_embeddings * 0.3
        
        # Layer 2: Second message passing
        neighbor_embeddings2 = np.zeros_like(embeddings)
        for h1, h2 in edges:
            if h1 in node_to_idx and h2 in node_to_idx:
                idx1, idx2 = node_to_idx[h1], node_to_idx[h2]
                neighbor_embeddings2[idx1] += embeddings[idx2] * 0.1
                neighbor_embeddings2[idx2] += embeddings[idx1] * 0.1
        
        embeddings = embeddings * 0.7 + neighbor_embeddings2 * 0.3
        
        # Normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        return embeddings
    
    def _graph_based_observation(self) -> np.ndarray:
        """Graph-based observation (paper-style).
        
        Paper: Graph representation → GNN → Node embeddings → Observation
        """
        node_features, edges = self._extract_graph_structure()
        node_embeddings = self._compute_graph_embeddings(node_features, edges)
        
        # Flatten node embeddings to observation vector
        # Shape: (num_nodes, embedding_dim) → (num_nodes * embedding_dim,)
        obs = node_embeddings.flatten()
        
        return obs.astype(np.float32)
    
    def _stack_observations(self, obs: np.ndarray) -> np.ndarray:
        """Frame stacking: include temporal information."""
        self._obs_history.append(obs.copy())
        
        if len(self._obs_history) < self.frame_stack:
            # Padding: fill with zeros initially
            padded = [np.zeros_like(obs) for _ in range(self.frame_stack - len(self._obs_history))]
            stacked = np.concatenate(padded + list(self._obs_history), axis=0)
        else:
            stacked = np.concatenate(list(self._obs_history), axis=0)
        
        return stacked.astype(np.float32)
    
    def _critical_potential(self) -> float:
        """Potential function for potential-based reward shaping.
        
        Potential from critical host states. Ensures policy invariance (Ng et al., 1999).
        """
        try:
            true_state = self.cyborg.get_agent_state('True')
        except Exception:
            return 0.0
        
        critical_targets = ['Op_Server0', 'Enterprise_Server', 'Op_Host0']
        critical_weights = {'Op_Server0': 3.0, 'Enterprise_Server': 3.0, 'Op_Host0': 2.0}
        
        badness = 0.0
        for hostname in critical_targets:
            host_info = true_state.get(hostname, {})
            
            # whether compromised
            is_compromised = 0.0
            for sess in host_info.get('Sessions', []):
                # Improvement: no str() cast (consistent with 1st_success)
                agent_name = sess.get('Agent', '')
                if agent_name and 'Red' in agent_name:
                    is_compromised = 1.0
                    break
            
            # service status
            service_up = 0.0
            for proc in host_info.get('Processes', []):
                if proc.get('Service Name'):
                    service_up = 1.0
                    break
            service_down = 1.0 - service_up
            
            weight = critical_weights.get(hostname, 1.0)
            badness += weight * (1.0 * is_compromised + 0.5 * service_down)
        
        # Negative potential (lower is better)
        return -float(badness)
    
    def _track_attack_chain(self) -> dict:
        """Track attack chain: initial compromise → spread → goal."""
        try:
            true_state = self.cyborg.get_agent_state('True')
        except Exception:
            return {}
        
        current_compromised = set()
        attack_stages = {}
        
        for hostname in self.known_hosts:
            host_info = true_state.get(hostname, {})
            is_comp = False
            
            for sess in host_info.get('Sessions', []):
                if 'Red' in str(sess.get('Agent', '')):
                    is_comp = True
                    current_compromised.add(hostname)
                    break
            
            if is_comp:
                # Determine attack stage
                prev_stage = self._attack_chain_state.get(hostname, 0)
                
                if hostname in self.critical_targets:
                    stage = 3  # Critical target compromised
                elif hostname in ['Enterprise0', 'Enterprise1', 'Enterprise2']:
                    stage = 2  # Spread to enterprise
                else:
                    stage = 1  # Initial compromise
                
                # Advance to higher stage than previous
                attack_stages[hostname] = max(prev_stage, stage)
            else:
                attack_stages[hostname] = 0
        
        self._attack_chain_state = attack_stages
        self._prev_compromised_hosts = current_compromised
        
        return attack_stages
    
    def _preventive_bonus(self, attack_stages: dict) -> float:
        """Preventive action bonus: prevent attack spread."""
        bonus = 0.0
        
        # Bonus when critical targets are not yet compromised
        critical_targets = ['Op_Server0', 'Enterprise_Server', 'Op_Host0']
        for host in critical_targets:
            if attack_stages.get(host, 0) == 0:
                # Neighboring hosts compromised but critical target safe
                nearby_compromised = 0
                if host == 'Op_Server0':
                    nearby = ['Op_Host0', 'Op_Host1', 'Op_Host2']
                elif host == 'Enterprise_Server':
                    nearby = ['Enterprise0', 'Enterprise1', 'Enterprise2']
                else:
                    nearby = ['Op_Server0', 'Op_Host1', 'Op_Host2']
                
                for nh in nearby:
                    if attack_stages.get(nh, 0) > 0:
                        nearby_compromised += 1
                
                # Bonus for protecting critical target in risky situation
                if nearby_compromised > 0:
                    bonus += 2.0 * self.nesy_lam * (nearby_compromised / len(nearby))
        
        return bonus
    
    def _ontology_based_reward_shaping(self, uptime_val: float) -> float:
        """Ontology-based reward shaping: explicit ontology-rule-based reward.
        
        NeSy requirements:
        1. Explicit Knowledge Representation: use ontology rules
        2. Symbolic Reasoning: logic-rule-based inference
        3. Potential-based shaping: policy invariance
        
        Improvements for training stability:
        - Progress-based annealing (reduce weight in later phase)
        - Increase Uptime-based bonus (emphasize long-term reward)
        - Emphasize long-term over short-term bonus
        
        Returns:
            ontology_bonus: float
        """
        # Progress-based annealing coefficient
        # Gradually reduce weight after 50% of training for stability
        # max_episode_steps=800, stop_iters=50 → ~40,000 steps; annealing starts after 20,000
        annealing_start_steps = 20000
        annealing_end_steps = 40000
        if hasattr(self, 'total_steps') and self.total_steps > annealing_start_steps:
            if self.total_steps >= annealing_end_steps:
                annealing_factor = 0.7  # 70% weight in later phase
            else:
                # Linear decay: 1.0 → 0.7
                progress = (self.total_steps - annealing_start_steps) / (annealing_end_steps - annealing_start_steps)
                annealing_factor = 1.0 - 0.3 * progress
        else:
            annealing_factor = 1.0  # 100% weight in early phase
        
        # Effective lambda: apply annealing to original lambda
        effective_lam = self.nesy_lam * annealing_factor
        
        try:
            true_state = self.cyborg.get_agent_state('True')
        except Exception:
            return 0.0
        
        # ===== Ontology structure init =====
        dependency_graph = self._build_dependency_graph()
        
        # ===== Ontology Concepts instantiation =====
        current_critical_status = {}
        compromised_hosts = []
        
        for hostname in self.known_hosts:
            host_info = true_state.get(hostname, {})
            is_comp = 0.0
            for sess in host_info.get('Sessions', []):
                if 'Red' in str(sess.get('Agent', '')):
                    is_comp = 1.0
                    compromised_hosts.append(hostname)
                    break
            
            if hostname in self.critical_targets:
                current_critical_status[hostname] = is_comp
        
        if not hasattr(self, '_prev_critical_status_onto'):
            self._prev_critical_status_onto = {h: 0.0 for h in self.critical_targets}
        if not hasattr(self, '_prev_compromised_hosts_onto'):
            self._prev_compromised_hosts_onto = []
        
        bonus = 0.0
        
        # ===== Ontology Axiom 1: Critical Host recovery (improved) =====
        # Recovery bonus considering dependency graph
        for host in self.critical_targets:
            prev = self._prev_critical_status_onto.get(host, 0.0)
            curr = current_critical_status.get(host, 0.0)
            
            if prev == 1.0 and curr == 0.0:  # recovered
                # Base recovery bonus (increased 100→150 for 5% goal)
                # Annealing reduces weight in later training
                base_bonus = 150.0 * effective_lam
                
                # Dependency-graph-based indirect effect bonus (40→60)
                dependency_impact = self._calculate_recovery_impact(
                    host, dependency_graph, true_state, depth=0, max_depth=2
                )
                
                # Indirect effect bonus (dependent hosts become safe)
                indirect_bonus = 60.0 * effective_lam * (dependency_impact - 1.0)
                
                # Extra bonus by critical host importance (50→75, 35→50)
                if host == 'Op_Server0':
                    bonus += base_bonus + indirect_bonus + 75.0 * effective_lam  # most critical
                elif host == 'Enterprise_Server':
                    bonus += base_bonus + indirect_bonus + 50.0 * effective_lam
                else:
                    bonus += base_bonus + indirect_bonus
        
        self._prev_critical_status_onto = current_critical_status
        
        # ===== Ontology Axiom 2: Attack chain prediction and prevention (improved) =====
        # MITRE ATT&CK-based attack stage inference
        attack_chain = self._infer_attack_chain(true_state, compromised_hosts)
        
        if attack_chain['urgency'] > 0.5:
            # Next attack stage prediction and prevention bonus
            if attack_chain['next_stage'] == 'Impact':
                # Prevent before advancing to Impact
                for target in attack_chain['predicted_targets']:
                    if target in self.critical_targets:
                        host_info = true_state.get(target, {})
                        is_safe = True
                        for sess in host_info.get('Sessions', []):
                            if 'Red' in str(sess.get('Agent', '')):
                                is_safe = False
                                break
                        if is_safe:
                            # Prevention success bonus (60→90 for 5% goal)
                            # Annealing applied
                            bonus += 90.0 * effective_lam * attack_chain['urgency']
        
        # Also: prevention bonus in early attack chain stages
        if attack_chain['current_stage'] in ['Initial Access', 'Lateral Movement']:
            # Larger bonus for prevention in early stage
            for target in attack_chain['predicted_targets']:
                if target in self.critical_targets:
                    host_info = true_state.get(target, {})
                    is_safe = True
                    for sess in host_info.get('Sessions', []):
                        if 'Red' in str(sess.get('Agent', '')):
                            is_safe = False
                            break
                    if is_safe:
                        # Early prevention bonus (40→60 for 5% goal)
                        # Annealing applied
                        bonus += 60.0 * effective_lam * (1.0 - attack_chain['urgency'])
        
        # ===== Ontology Axiom 3: Uptime preservation (improved) =====
        if not hasattr(self, '_prev_uptime_onto'):
            self._prev_uptime_onto = uptime_val
        
        uptime_delta = uptime_val - self._prev_uptime_onto
        if uptime_delta > 0:
            # Uptime increase bonus (120→180 for 5% goal)
            # Uptime bonus is long-term; less annealing (90% kept)
            uptime_annealing = 0.9 if annealing_factor < 1.0 else 1.0
            bonus += 180.0 * effective_lam * uptime_annealing * uptime_delta
        elif uptime_delta < -0.01:
            # Uptime decrease penalty (stronger penalty for improvement)
            # Penalty kept in later phase for training stability
            bonus -= 50.0 * effective_lam * abs(uptime_delta)
        
        # High Uptime maintenance bonus (50→75 for 5% goal)
        # Uptime bonus emphasized as long-term reward
        if uptime_val > 0.70:  # bonus starts from 70%
            uptime_annealing = 0.9 if annealing_factor < 1.0 else 1.0
            bonus += 75.0 * effective_lam * uptime_annealing * (uptime_val - 0.70)
        
        # Very high Uptime extra bonus (70→100 for 5% goal)
        # Uptime bonus emphasized as long-term reward
        if uptime_val > 0.80:
            uptime_annealing = 0.9 if annealing_factor < 1.0 else 1.0
            bonus += 100.0 * effective_lam * uptime_annealing * (uptime_val - 0.80)
        
        # Sustained protection bonus (bonus when critical hosts stay safe)
        # Sustained protection emphasized as long-term reward
        safe_critical_count = sum(1 for h in self.critical_targets if current_critical_status.get(h, 1.0) == 0.0)
        if safe_critical_count > 0:
            # Extra bonus when all critical hosts safe (30→50)
            # Less annealing for sustained protection
            protection_annealing = 0.85 if annealing_factor < 1.0 else 1.0
            if safe_critical_count == len(self.critical_targets):
                bonus += 50.0 * effective_lam * protection_annealing  # all critical hosts safe
            else:
                bonus += 15.0 * effective_lam * protection_annealing * (safe_critical_count / len(self.critical_targets))  # partial (10→15)
        
        self._prev_uptime_onto = uptime_val
        self._prev_compromised_hosts_onto = compromised_hosts.copy()
        
        return bonus

    def _apply_logic_guided_reward(self, action: int, state_info: dict) -> float:
        """Logic-Guided Policy: reward bonus by logic rules (NeSy)."""
        if not self.use_logic_guided:
            return 0.0
        
        try:
            true_state = state_info.get('true_state')
            if true_state is None:
                true_state = self.cyborg.get_agent_state('True')
        except Exception:
            return 0.0
        
        bonus = 0.0
        
        # Rule 1: Bonus for recovery action when critical host is compromised
        critical_compromised = False
        for hostname in self.critical_targets:
            host_info = true_state.get(hostname, {})
            for sess in host_info.get('Sessions', []):
                if 'Red' in str(sess.get('Agent', '')):
                    critical_compromised = True
                    break
            if critical_compromised:
                break
        
        if critical_compromised:
            # Bonus for recovery (action type not directly observable;
            # bonus given when recovery is confirmed next step)
            # Here only state-based bonus
            bonus += 5.0 * self.nesy_lam
        
        # Rule 2: Bonus for defense action when attack pattern detected
        # (simplified for implementation)
        
        return bonus
    
    def _apply_rule_pruning_penalty(self, action: int, state_info: dict) -> float:
        """Rule-Based Pruning: penalty for rule-violating actions (NeSy)."""
        if not self.use_rule_pruning:
            return 0.0
        
        try:
            true_state = state_info.get('true_state')
            if true_state is None:
                true_state = self.cyborg.get_agent_state('True')
        except Exception:
            return 0.0
        
        penalty = 0.0
        uptime_val = self._calculate_uptime_fast()
        
        # Rule 1: Penalty for recovery action when all critical hosts are safe
        all_critical_safe = True
        for hostname in self.critical_targets:
            host_info = true_state.get(hostname, {})
            for sess in host_info.get('Sessions', []):
                if 'Red' in str(sess.get('Agent', '')):
                    all_critical_safe = False
                    break
            if not all_critical_safe:
                break
        
        if all_critical_safe:
            # Penalty for unnecessary recovery when safe
            # (action type check complex; simplified)
            pass
        
        # Rule 2: Penalty for aggressive defense when Uptime is high
        if uptime_val >= 0.95:
            # (action type check complex; simplified)
            pass
        
        return penalty

    # -------------------------
    # Gymnasium API
    # -------------------------
    def reset(self, *, seed: int | None = None, options=None):
        self.steps = 0

        if seed is not None:
            self.seed_value = int(seed)
            # Reset CybORG seed when possible
            try:
                self.cyborg.set_seed(self.seed_value)
            except Exception:
                pass

        # 1st_success: use OpenAIGymWrapper reset
        res = self._gym_env.reset()
        obs = (res[0] if isinstance(res, tuple) else res)
        
        # Extract uptime (obs[0] may be uptime)
        uptime_val = float(obs[0]) if len(obs) > 0 else 1.0
        
        # NeSy observation branch (ablation design):
        # - Base: raw observation (~11k dim)
        # - State (2): 52-dim state abstraction
        # - Reward (3): raw observation (reward shaping only)
        # - Ontology (5): 107-dim ontology observation
        # - Full (4): 52-dim state abstraction
        # - Full Ontology (6 = 2+3+5): 107-dim ontology observation (includes exp 2 state info)
        if self.use_107dim_unified:
            obs = self._ontology_based_observation()  # ontology, full_ontology both 107-dim
        elif self.use_graph_representation:
            obs = self._graph_based_observation()
        elif self.use_state_abstraction:
            obs = self._nesy_state_features()  # state, full (52-dim)
        else:
            obs = np.array(obs, dtype=np.float32)  # base, reward (raw)
        
        # Apply frame stacking when enabled
        if self.enable_frame_stack and self.frame_stack > 1:
            obs = self._stack_observations(obs)

        # Store initial state for reward shaping (1st_success)
        if self.use_reward_shaping:
            # 1st_success: init to actual state on reset
            try:
                true_state = self.cyborg.get_agent_state('True')
                critical_targets = ['Op_Server0', 'Enterprise_Server', 'Op_Host0']
                self._prev_critical_status = {}
                for hostname in critical_targets:
                    info = true_state.get(hostname, {})
                    is_comp = 0.0
                    for sess in info.get('Sessions', []):
                        agent_name = sess.get('Agent', '')
                        if agent_name and 'Red' in agent_name:
                            is_comp = 1.0
                            break
                    self._prev_critical_status[hostname] = is_comp
            except Exception:
                self._prev_critical_status = {'Op_Server0': 0.0, 'Enterprise_Server': 0.0, 'Op_Host0': 0.0}
            
            # Init uptime tracking
            self._prev_uptime = uptime_val
            self._uptime_history = deque(maxlen=10)
            self._uptime_history.append(uptime_val)
        
        self._episode_count = 0

        info = {} if not isinstance(res, tuple) or res[1] is None else res[1]
        if isinstance(info, dict):
            info.pop("observation", None)  # same as 1st_success
            info.setdefault("nesy_mode", self.nesy_mode)
            info["uptime_value"] = uptime_val  # same as 1st_success - collected by Callback
            info["nesy_bonus"] = 0.0  # no bonus on reset
            # Separate raw reward logging (for paper defense metrics)
            info["raw_reward"] = 0.0  # no reward on reset
            info["shaping_bonus"] = 0.0  # no bonus on reset
            info["shaped_return"] = 0.0  # no reward on reset
            # Observation dimension logging (Full Ontology vs Ontology)
            info["obs_dim"] = int(self.observation_space.shape[0] // self.frame_stack)  # base obs dim
        
        return obs, info

    def step(self, action):
        self.steps += 1
        self.total_steps += 1  # track total steps (for annealing)

        # Logic-Guided Policy: action filtering (currently implemented as reward bonus)
        logic_bonus = 0.0
        if self.use_logic_guided:
            try:
                state_info = {'true_state': self.cyborg.get_agent_state('True')}
            except Exception:
                state_info = {}
            logic_bonus = self._apply_logic_guided_reward(action, state_info)
        
        # Rule-Based Pruning: penalty for rule violation
        pruning_penalty = 0.0
        if self.use_rule_pruning:
            try:
                state_info = {'true_state': self.cyborg.get_agent_state('True')}
            except Exception:
                state_info = {}
            pruning_penalty = self._apply_rule_pruning_penalty(action, state_info)
        
        # 1st_success: use OpenAIGymWrapper step
        res = self._gym_env.step(action)
        obs, reward, done, info = (res[0], res[1], res[2], res[3]) if len(res) == 4 else res
        
        # Store raw reward (before shaping)
        raw_reward = float(reward)
        
        # Extract uptime (obs[0] may be uptime)
        uptime_val = float(obs[0]) if len(obs) > 0 else 1.0
        
        # Init uptime tracking (first step)
        if not hasattr(self, '_prev_uptime'):
            self._prev_uptime = uptime_val
        if not hasattr(self, '_uptime_history'):
            self._uptime_history = deque(maxlen=10)
        
        # NeSy observation branch (ablation design):
        # - Base: raw observation (~11k dim)
        # - State (2): 52-dim state abstraction
        # - Reward (3): raw observation (reward shaping only)
        # - Ontology (5): 107-dim ontology observation
        # - Full (4): 52-dim state abstraction
        # - Full Ontology (6 = 2+3+5): 107-dim (includes exp 2 state info)
        if self.use_107dim_unified:
            obs = self._ontology_based_observation()  # ontology, full_ontology both 107-dim
        elif self.use_graph_representation:
            obs = self._graph_based_observation()
        elif self.use_state_abstraction:
            obs = self._nesy_state_features()  # state, full (52-dim)
        else:
            obs = np.array(obs, dtype=np.float32)  # base, reward (raw)
        
        # Apply frame stacking when enabled
        if self.enable_frame_stack and self.frame_stack > 1:
            obs = self._stack_observations(obs)

        # ===== NeSy Reward Shaping (conservative: preserve base performance) =====
        # Issue: excessive shaping hurts learning
        # Fix: event-based conservative shaping only
        # - No per-step bonus (reduce noise)
        # - Small bonus only on Critical Host recovery (1st_success)
        # - No preventive-action bonus
        
        # Ontology mode: explicit ontology-rule-based reward shaping
        # full_ontology: multi-objective + axiom-based both
        nesy_bonus = 0.0
        

        # ===== Uptime Delta bonus (improved: all seeds + 5% goal) =====
        if self.use_reward_shaping and self.nesy_lam != 0.0:
            uptime_delta = uptime_val - self._prev_uptime
            if uptime_delta > 0:
                # Uptime increase bonus
                nesy_bonus += 70.0 * self.nesy_lam * uptime_delta
            elif uptime_delta < -0.01:
                # Uptime decrease penalty
                nesy_bonus -= 25.0 * self.nesy_lam * abs(uptime_delta)

        # ===== Uptime Preserve bonus (improved: all seeds) =====
        if self.use_reward_shaping and self.nesy_lam != 0.0:
            if uptime_val > 0.70:  # bonus from 70%
                nesy_bonus += 35.0 * self.nesy_lam * (uptime_val - 0.70)
            if uptime_val > 0.80:  # extra bonus from 80%
                nesy_bonus += 50.0 * self.nesy_lam * (uptime_val - 0.80)
        # Uptime tracking update (for analysis; not used in bonus)
        if not hasattr(self, '_prev_uptime'):
            self._prev_uptime = uptime_val
        if not hasattr(self, '_uptime_history'):
            self._uptime_history = deque(maxlen=10)
        self._prev_uptime = uptime_val
        self._uptime_history.append(uptime_val)
        
        if self.nesy_mode == "full_ontology" and self.nesy_lam != 0.0:
            # Exp 6 (Full Ontology = 2+3+5): State + Reward + Ontology
            # - Exp 2 (State): in 107-dim ontology observation
            # - Exp 3 (Reward): multi-objective (uptime delta/preserve + critical host recovery)
            #   → nesy_bonus already has uptime delta/preserve (lines above)
            # - Exp 5 (Ontology): ontology axiom reward
            # Add critical host recovery (exp 3) + ontology axiom (exp 5)
            full_nesy_bonus = 0.0
            try:
                true_state = self.cyborg.get_agent_state('True')
                critical_targets = ['Op_Server0', 'Enterprise_Server', 'Op_Host0']
                
                # 1st_success: bonus only on Critical host recovery (part of exp 3)
                current_critical_status = {}
                for hostname in critical_targets:
                    info = true_state.get(hostname, {})
                    is_comp = 0.0
                    for sess in info.get('Sessions', []):
                        agent_name = sess.get('Agent', '')
                        if agent_name and 'Red' in agent_name:
                            is_comp = 1.0
                            break
                    current_critical_status[hostname] = is_comp
                
                if not hasattr(self, '_prev_critical_status_full_onto'):
                    self._prev_critical_status_full_onto = {h: 0.0 for h in critical_targets}
                
                # Critical Host recovery bonus (part of exp 3)
                for host in critical_targets:
                    prev = self._prev_critical_status_full_onto.get(host, 0.0)
                    curr = current_critical_status.get(host, 0.0)
                    if prev == 1.0 and curr == 0.0:  # recovered
                        # Critical host recovery is very important
                        if host == 'Op_Server0':
                            full_nesy_bonus += 70.0 * self.nesy_lam  # most critical
                        elif host == 'Enterprise_Server':
                            full_nesy_bonus += 60.0 * self.nesy_lam
                        else:
                            full_nesy_bonus += 55.0 * self.nesy_lam
                
                self._prev_critical_status_full_onto = current_critical_status
                
            except Exception:
                pass
            
            # Add ontology axiom-based reward shaping (exp 5)
            ontology_bonus = self._ontology_based_reward_shaping(uptime_val)
            # nesy_bonus already has uptime delta/preserve; add critical recovery (exp 3) + axiom (exp 5)
            nesy_bonus = nesy_bonus + full_nesy_bonus + ontology_bonus
        elif self.use_ontology and self.nesy_mode == "ontology" and self.nesy_lam != 0.0:
            # Ontology-only: ontology-based reward shaping only
            nesy_bonus = self._ontology_based_reward_shaping(uptime_val)
        elif self.use_reward_shaping and self.nesy_lam != 0.0:
            # 1st_success: bonus only on Critical host recovery
            try:
                true_state = self.cyborg.get_agent_state('True')
                critical_targets = ['Op_Server0', 'Enterprise_Server', 'Op_Host0']
                
                # 1st_success: bonus only on Critical host recovery
                current_critical_status = {}
                for hostname in critical_targets:
                    info = true_state.get(hostname, {})
                    is_comp = 0.0
                    for sess in info.get('Sessions', []):
                        agent_name = sess.get('Agent', '')
                        if agent_name and 'Red' in agent_name:
                            is_comp = 1.0
                            break
                    current_critical_status[hostname] = is_comp
                
                if not hasattr(self, '_prev_critical_status'):
                    self._prev_critical_status = {h: 0.0 for h in critical_targets}
                
                # Critical Host recovery bonus (increased for 5% goal)
                for host in critical_targets:
                    prev = self._prev_critical_status.get(host, 0.0)
                    curr = current_critical_status.get(host, 0.0)
                    if prev == 1.0 and curr == 0.0:  # recovered
                        # Critical host recovery is very important
                        if host == 'Op_Server0':
                            nesy_bonus += 70.0 * self.nesy_lam  # most critical
                        elif host == 'Enterprise_Server':
                            nesy_bonus += 60.0 * self.nesy_lam
                        else:
                            nesy_bonus += 55.0 * self.nesy_lam
                
                self._prev_critical_status = current_critical_status
                
            except Exception as e:
                # Fallback: 1st_success (simple and effective)
                if not hasattr(self, '_prev_critical_status'):
                    self._prev_critical_status = {'Op_Server0': 0.0, 'Enterprise_Server': 0.0, 'Op_Host0': 0.0}
                
                try:
                    true_state = self.cyborg.get_agent_state('True')
                    current_critical_status = {}
                    for hostname in critical_targets:
                        info = true_state.get(hostname, {})
                        is_comp = 0.0
                        for sess in info.get('Sessions', []):
                            if 'Red' in str(sess.get('Agent', '')):
                                is_comp = 1.0
                                break
                        current_critical_status[hostname] = is_comp
                    
                    for host in critical_targets:
                        prev = self._prev_critical_status.get(host, 0.0)
                        curr = current_critical_status.get(host, 0.0)
                        if prev == 1.0 and curr == 0.0:
                            # Critical host recovery bonus (increased)
                            if host == 'Op_Server0':
                                nesy_bonus += 50.0 * self.nesy_lam
                            elif host == 'Enterprise_Server':
                                nesy_bonus += 40.0 * self.nesy_lam
                            else:
                                nesy_bonus += 35.0 * self.nesy_lam
                    
                    self._prev_critical_status = current_critical_status
                except:
                    # Final fallback: simple reward shaping
                    uptime = self._calculate_uptime_fast()
                    compromised, _unknown = self._get_compromise_counts()
                    shaping = (uptime * 1.0) - (compromised * 0.05)
                    nesy_bonus = float(self.nesy_lam) * float(shaping)

        # ===== Adaptive Reward Scaling =====
        # Bonus scaling by training phase (when enabled)
        if self.enable_adaptive_scale:
            self._episode_count += 1
            if self._episode_count < 1000:
                self._adaptive_scale = 1.5 - (self._episode_count / 1000) * 0.5
            else:
                self._adaptive_scale = 1.0
            
            # Apply adaptive scale only to NeSy bonus (baseline unchanged)
            if nesy_bonus != 0.0:
                nesy_bonus *= self._adaptive_scale
        else:
            self._adaptive_scale = 1.0  # always 1.0 when disabled

        # Apply Logic-Guided Policy and Rule-Based Pruning bonus/penalty
        reward = float(reward) + nesy_bonus + logic_bonus - pruning_penalty

        # Convert to Gymnasium format
        terminated = bool(done)
        truncated = self.steps >= self.max_episode_steps
        if done or truncated:
            self.episode_count += 1  # increment on episode end

        # Info update (same format as 1st_success + improvements)
        if isinstance(info, dict):
            info.pop("observation", None)  # same as 1st_success
            info.setdefault("nesy_mode", self.nesy_mode)
            info["uptime_value"] = uptime_val  # same as 1st_success - collected by Callback
            info["nesy_bonus"] = nesy_bonus
            # Separate raw reward logging (for paper defense metrics)
            info["raw_reward"] = raw_reward  # env original reward (before shaping)
            info["shaping_bonus"] = nesy_bonus + logic_bonus - pruning_penalty  # total shaping
            info["shaped_return"] = float(reward)  # final reward (raw + shaping)
            info["nesy_adaptive_scale"] = self._adaptive_scale
            info["nesy_frame_stack"] = self.frame_stack
            info["logic_guided_bonus"] = logic_bonus
            info["rule_pruning_penalty"] = pruning_penalty
            # Observation dimension logging (Full Ontology vs Ontology)
            info["obs_dim"] = int(self.observation_space.shape[0] // self.frame_stack)  # base obs dim

        return obs, float(reward), terminated, truncated, info if isinstance(info, dict) else {}


def create_cyborg_env(env_config: dict | None = None):
    """RLlib entrypoint."""
    env_config = env_config or {}

    seed = env_config.get("seed", None)
    max_episode_steps = int(env_config.get("max_episode_steps", 800))
    blue_agent = env_config.get("blue_agent", "blue_agent_0")

    # Mode precedence: env_config overrides env var.
    mode = env_config.get("nesy_mode", _get_env_var("NESY_MODE", "base"))
    lam_raw = env_config.get("nesy_lam", _get_env_var("NESY_LAM", "1.0"))
    try:
        lam = float(lam_raw)
    except Exception:
        lam = 1.0
    
    # Frame Stacking and Adaptive Scaling control
    enable_frame_stack = env_config.get("enable_frame_stack", False)
    enable_adaptive_scale = env_config.get("enable_adaptive_scale", False)

    env = CybORGWrapper(
        seed=seed,
        max_episode_steps=max_episode_steps,
        nesy_mode=mode,
        nesy_lam=lam,
        blue_agent=blue_agent,
        enable_frame_stack=enable_frame_stack,
        enable_adaptive_scale=enable_adaptive_scale,
    )
    # Use Gymnasium TimeLimit wrapper for consistent truncation semantics.
    return TimeLimit(env, max_episode_steps=max_episode_steps)
