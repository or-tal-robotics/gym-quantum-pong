from gym.envs.registration import register

register(
    id='Quantum_Pong-v0',
    entry_point='gym_quantum_pong.envs:QuantumPongEnv',
)