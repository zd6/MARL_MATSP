from gym.envs.registration import register
register(
    id = 'matsp-v0',
    entry_point = 'matsp.envs:GridEnv'
)