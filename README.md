# Gamma and Vega Hedging Using Deep Distributional Reinforcement Learning

## About

This is the companion code for the paper *Gamma and Vega Hedging Using Deep Distributional Reinforcement Learning
* by Jay Cao, Jacky Chen, Soroush Farghadani, John Hull, Zissis Poulos, Zeyu Wang and Jun Yuan. The paper is available [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4106814) at SSRN.

## Code Structure
```
Gamma & Vega Hedging Codebase
│   run.py - Run D4PG model
│   greek_run.py - Run baseline models
|   simulate_env.py - Simulate the environment
└───agent
│   │   agent.py - D4PG agent
│   │   distributional.py - distributional dependency for D4PG
│   │   learning.py - learning module for D4PG
└───environment
│   │   Environment.py - Trading Environment
│   │   Trading.py - Portfolio constructions
│   │   utils.py - Stochastic Processes generation and other utility functions
└───Result Analysis
    │   RL&Baseline Result Analysis - Sample RL and Baseline model result analysis
└───Logs
    |   Sample Log Files - This is NOT the results in the paper! Just a sample log file with 100 train_simulations and 100 evaluation epochs.
```

## Dependencies
```
dm-env==1.5
gym==0.24.1
numpy==1.23.1
pytest==6.2.5
reverb==2.0.1
scipy==1.8.1
torch==1.11.0
tqdm==4.64.0
dm-acme[jax,tensorflow,envs]==0.4.0
dm-sonnet==2.0.0
dm-launchpad==0.5.0
trfl==1.2.0
pyyaml==5.4.1
xmanager==0.2.0
```

## Sample Runs

#### *feed_data* parameter added (feed_data=True: feed real data into the trained model for evaluation)

### 1. Train & Evaluate the Reinforcement Learning Agent

##### RL, gbm, feed_data=True
```
python run.py -spread=0.005 -obj_func=meanstd -train_sim=40000 -critic=qr-gl -std_coef=1.645 -feed_data=True -init_vol=0.3 -mu=0.0 -vov=0.0 -vega_obs=False -gbm=True -hed_ttm=30 -liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -action_space=0,1 -logger_prefix=batch1/Table1/TC05/RL/meanstd -n_step=5
```

##### RL, gbm, feed_data=False (eval_sim needed)
```
python run.py -spread=0.005 -obj_func=meanstd -train_sim=40000 -eval_sim=5000 -critic=qr-gl -std_coef=1.645 -feed_data=True -init_vol=0.3 -mu=0.0 -vov=0.0 -vega_obs=False -gbm=True -hed_ttm=30 -liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -action_space=0,1 -logger_prefix=batch1/Table1/TC05/RL/meanstd -n_step=5
```

##### RL, sabr, feed_data=True
```
python run.py -spread=0.005 -obj_func=meanstd -train_sim=40000 -critic=qr-gl -std_coef=1.645 -feed_data=True -init_vol=0.3 -mu=0.0 -vov=0.0 -vega_obs=True -sabr=True -hed_ttm=30 -liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -action_space=0,1 -logger_prefix=batch1/Table3/TC05/RL/meanstd -n_step=5
```

##### RL, sabr, feed_data=False (eval_sim needed)
```
python run.py -spread=0.005 -obj_func=meanstd -train_sim=40000 -eval_sim=5000 -critic=qr-gl -std_coef=1.645 -feed_data=True -init_vol=0.3 -mu=0.0 -vov=0.0 -vega_obs=True -sabr=True -hed_ttm=30 -liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -action_space=0,1 -logger_prefix=batch1/Table3/TC05/RL/meanstd -n_step=5
```

### 2. Evaluate a Baseline Agent

##### Delta, gbm, feed_data=True
```
python greek_run.py -spread=0.005 -gbm=True -strategy=delta -feed_data=True -init_vol=0.3 -mu=0.0 -vov=0.0 -hed_ttm=30 -liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -vega_obs=False -logger_prefix=batch1/Table1/TC05/Baseline/delta
```

##### Delta_Vega, gbm, feed_data=False (eval_sim needed)
```
python greek_run.py -spread=0.005 -gbm=True -eval_sim=5000 -strategy=vega -feed_data=False -init_vol=0.3 -mu=0.0 -vov=0.3 -hed_ttm=30 -liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -vega_obs=True -logger_prefix=batch1/Table3/TC05/Baseline/vega
```

##### Delta, sabr, feed_data=True
```
python greek_run.py -spread=0.005 -sabr=True -strategy=delta -feed_data=True -init_vol=0.3 -mu=0.0 -vov=0.3 -hed_ttm=30 -liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -vega_obs=True -logger_prefix=batch1/Table3/TC05/Baseline/delta
```

##### Delta_Vega, sabr, feed_data=False (eval_sim needed)
```
python greek_run.py -spread=0.005 -sabr=True -eval_sim=5000 -strategy=vega -feed_data=False -init_vol=0.3 -mu=0.0 -vov=0.3 -hed_ttm=30 -liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -vega_obs=True -logger_prefix=batch1/Table3/TC05/Baseline/vega
```



## Result Log Files

Trained and Tested Logs are stored in the `Logs` folder.

## Credits

* The implementation of D4PG agent is taken from ACME [D4PG](https://github.com/deepmind/acme/tree/master/acme/agents/tf/d4pg).
