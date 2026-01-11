# cyborg_env_maker.py
# [NeSY v2: MITRE ATT&CK Knowledge & Potential-based Shaping]
#
# 설계:
# 1. State Abstraction: 네트워크 상태를 MITRE ATT&CK 전술(Tactic) 단계로 요약
# 2. Potential Shaping: 보안 상태 점수(Phi)의 변화량(Delta)으로 보상을 유도 (Optimal Policy 보존)
# 3. Compatibility: Gym vs Gymnasium 충돌 방지 (Duck Typing 적용)

from typing import Optional, Dict, Any, Tuple, Union, List
import os
import math
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import EnvSpec

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper

def _get_nesy_params() -> float:
    """NESY_LAM: 0.0=Baseline, >0.0=NeSY (Shaping Strength)"""
    v = os.environ.get("NESY_LAM", None)
    if v is None: return 0.0
    try: return float(v)
    except ValueError: return 0.0

class _CybORGBlueGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None, blue_agent="blue_agent_0", max_episode_steps=800):
        super().__init__()
        self._seed = int(seed) if seed is not None else None
        self._blue_agent = blue_agent
        self._max_episode_steps = int(max_episode_steps)
        
        self.scenario = DroneSwarmScenarioGenerator()
        self.cyborg = CybORG(self.scenario, "sim")
        
        # Baseline용 Raw Wrapper
        self.raw_wrapped = FixedFlatWrapper(self.cyborg)
        self._gym_env = OpenAIGymWrapper(agent_name=self._blue_agent, env=self.raw_wrapped)
        
        self.spec = EnvSpec(id="CybORG-MITRE-NeSY-v2", max_episode_steps=self._max_episode_steps)

        # [Topology Knowledge] 호스트 정의
        self.known_hosts = [
            'User0', 'User1', 'User2', 'User3', 'User4',
            'Enterprise0', 'Enterprise1', 'Enterprise2', 'Enterprise_Server',
            'Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2',
            'Defender', 'User_Router', 'Enterprise_Router', 'Op_Router'
        ]
        # 중요 자산 (Impact 단계의 타겟)
        self.critical_targets = ['Op_Server0', 'Enterprise_Server', 'Op_Host0']
        
        # 호스트 중요도 (Criticality) 맵핑
        self.host_criticality = {}
        for h in self.known_hosts:
            if h in self.critical_targets:
                self.host_criticality[h] = 1.0 # 매우 중요
            elif 'Op_' in h:
                self.host_criticality[h] = 0.7 # 운영망 중요
            elif 'Enterprise' in h:
                self.host_criticality[h] = 0.5 # 기업망 중간
            else:
                self.host_criticality[h] = 0.2 # 사용자망 낮음

        # [NeSY State Definition]
        # 1. Global Uptime (1)
        # 2. Tactic Scores (4): Recon, Access, Lateral, Impact
        # 3. Host Risks (17): 각 호스트별 위험도 * 중요도
        # 총 차원 = 1 + 4 + 17 = 22D
        self.nesy_dim = 1 + 4 + len(self.known_hosts)
        
        # [모드 확인] Baseline vs NeSY
        self.lam = _get_nesy_params()
        self.is_nesy = (self.lam > 0.0)
        
        # Observation Space 설정
        if self.is_nesy:
            # NeSY: 22차원 지식 벡터
            self._final_obs_dim = self.nesy_dim
        else:
            # Baseline: 11k Raw Flat Obs (에러 방지를 위해 shape 직접 확인)
            base_obs_space = self._gym_env.observation_space
            if hasattr(base_obs_space, "shape"):
                self._final_obs_dim = int(base_obs_space.shape[0])
            else:
                self._final_obs_dim = 11293 # Fallback
            
        low = np.full((self._final_obs_dim,), -np.inf, dtype=np.float32)
        high = np.full((self._final_obs_dim,), np.inf, dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        
        # Action Space
        if hasattr(self._gym_env.action_space, "n"):
            n_actions = int(self._gym_env.action_space.n)
        else:
            n_actions = 56 
        self.action_space = Discrete(n_actions)
        
        self._np_random, _ = gym.utils.seeding.np_random(self._seed)
        
        # Potential Shaping을 위한 이전 상태 저장
        self._prev_potential = 0.0

    # ----------------------------------------------------------
    # MITRE ATT&CK Logic (Knowledge Layer)
    # ----------------------------------------------------------
    def _analyze_mitre_state(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        TrueState를 분석하여 MITRE 전술 점수와 Potential(Phi)을 계산합니다.
        """
        true_state = self.cyborg.get_agent_state('True')
        
        # Tactic 별 카운터
        recon_cnt = 0
        access_cnt = 0
        lateral_cnt = 0
        impact_cnt = 0
        
        host_risks = []
        
        for hostname in self.known_hosts:
            info = true_state.get(hostname, {})
            
            # 호스트별 위험도 분석
            is_compromised = False
            for sess in info.get('Sessions', []):
                if 'Red' in sess.get('Agent', ''):
                    is_compromised = True
                    break
            
            risk_val = 1.0 if is_compromised else 0.0
            
            # Criticality 반영: Risk x Criticality
            weighted_risk = risk_val * self.host_criticality.get(hostname, 0.0)
            host_risks.append(weighted_risk)
            
            if is_compromised:
                # Tactic Mapping
                if hostname in self.critical_targets:
                    impact_cnt += 1 # Impact
                elif 'Op_' in hostname or 'Router' in hostname:
                    lateral_cnt += 1 # Lateral Movement
                else:
                    access_cnt += 1 # Initial Access
                
                recon_cnt += 1 
        
        # Tactic Scores (Normalize)
        s_recon = math.tanh(recon_cnt / 5.0)
        s_access = math.tanh(access_cnt / 3.0)
        s_lateral = math.tanh(lateral_cnt / 2.0)
        s_impact = math.tanh(impact_cnt / 1.0)
        
        tactic_vec = np.array([s_recon, s_access, s_lateral, s_impact], dtype=np.float32)
        risk_vec = np.array(host_risks, dtype=np.float32)
        
        # Potential Function (Phi)
        # Phi = 1.0 - Risk (공격 단계가 깊을수록 Risk가 큼)
        risk_score = (0.5 * s_impact) + (0.3 * s_lateral) + (0.15 * s_access) + (0.05 * s_recon)
        potential = 1.0 - risk_score 
        
        return tactic_vec, risk_vec, potential

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = int(seed)
        res = self._gym_env.reset()
        obs = (res[0] if isinstance(res, tuple) else res)
        uptime_val = float(obs[0]) if len(obs) > 0 else 1.0
        
        # 초기 상태 분석
        tactic_vec, risk_vec, phi = self._analyze_mitre_state()
        self._prev_potential = phi
        
        if self.is_nesy:
            # NeSY: [Uptime, Tactics, HostRisks]
            final_obs = np.concatenate([
                np.array([uptime_val]), tactic_vec, risk_vec
            ]).astype(np.float32)
        else:
            # Baseline: Raw Flat Obs
            final_obs = np.array(obs, dtype=np.float32)
        
        info = {} if not isinstance(res, tuple) or res[1] is None else res[1]
        if isinstance(info, dict):
            info.pop("observation", None)
            info["uptime_value"] = uptime_val
            info["nesy_bonus"] = 0.0
        return final_obs, info

    def step(self, action):
        res = self._gym_env.step(action)
        obs, reward, done, info = (res[0], res[1], res[2], res[3]) if len(res) == 4 else res
        uptime_val = float(obs[0]) if len(obs) > 0 else 1.0
        
        # 상태 분석
        tactic_vec, risk_vec, phi_now = self._analyze_mitre_state()
        
        if self.is_nesy:
            final_obs = np.concatenate([
                np.array([uptime_val]), tactic_vec, risk_vec
            ]).astype(np.float32)
        else:
            final_obs = np.array(obs, dtype=np.float32)
        
        # ----------------------------------------------------------
        # Potential-based Reward Shaping (NeSY Only)
        # F = gamma * Phi(s') - Phi(s)
        # ----------------------------------------------------------
        nesy_bonus = 0.0
        if self.is_nesy:
            gamma = 0.99 
            # Potential의 변화량(Delta)을 보상으로 사용
            shaping = gamma * phi_now - self._prev_potential
            # Scaling (Shaping 강도)
            nesy_bonus = shaping * 10.0 * self.lam
            
        self._prev_potential = phi_now
        
        # 최종 보상
        total_reward = float(reward) + nesy_bonus

        if isinstance(info, dict):
            info.pop("observation", None)
            info["uptime_value"] = uptime_val
            info["nesy_bonus"] = nesy_bonus

        return final_obs, float(total_reward), bool(done), False, info
        
    def close(self):
        self._gym_env.close()

def create_cyborg_env(seed=0, max_episode_steps=800, blue_agent="blue_agent_0"):
    env = _CybORGBlueGymEnv(seed=seed, blue_agent=blue_agent, max_episode_steps=max_episode_steps)
    return TimeLimit(env, max_episode_steps=max_episode_steps)

make_cyborg_env = create_cyborg_env