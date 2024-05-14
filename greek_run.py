import os
from pathlib import Path

import acme
from acme import wrappers
import acme.utils.loggers as log_utils
import dm_env

from environment.Environment import TradingEnv
from environment.utils import Utils
from agent.agent import DeltaHedgeAgent, GammaHedgeAgent, VegaHedgeAgent

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('eval_sim', 5_000, 'evaluation episodes (Default 40_000)')
flags.DEFINE_integer('init_ttm', 60, 'number of days in one episode (Default 60)')
flags.DEFINE_float('mu', 0.0, 'spot drift (Default 0.2)')
flags.DEFINE_float('init_vol', 0.2, 'initial spot vol (Default 0.2)')
flags.DEFINE_float('poisson_rate', 1.0, 'possion rate of new optiosn in liability portfolio (Default 1.0)')
flags.DEFINE_float('moneyness_mean', 1.0, 'new optiosn moneyness mean (Default 1.0)')
flags.DEFINE_float('moneyness_std', 0.0, 'new optiosn moneyness std (Default 0.0)')
flags.DEFINE_float('spread', 0.0, 'Hedging transaction cost (Default 0.0)')
flags.DEFINE_string('strategy', 'delta', 'Hedging strategy opt: delta / gamma/ vega (Default delta')
flags.DEFINE_float('vov', 0.0, 'Vol of vol, zero means BSM; non-zero means SABR (Default 0.0)')
flags.DEFINE_list('liab_ttms',['60',], 'List of maturities selected for new adding option (Default [60,])')
flags.DEFINE_integer('hed_ttm', 20, 'Hedging option maturity in days (Default 20)')
flags.DEFINE_string('logger_prefix', '', 'Prefix folder for logger (Default None)')
flags.DEFINE_boolean('vega_obs', False, 'Include portfolio vega and hedging option vega in state variables (Default False)')
flags.DEFINE_boolean('gbm', False, 'GBM (Default False)')
flags.DEFINE_boolean('sabr', False, 'SABR (Default False)')
flags.DEFINE_integer('hed_frq', 1, 'Hedging frequency (Default 1): i.e. hed_frq=2 means hedging twice a day')
flags.DEFINE_boolean('feed_data', False, 'Feed real data into trained model for evaluation (Default False)')
flags.DEFINE_boolean('feed_data_fx', False, 'Feed real fx data into trained model for evaluation (Default False)')

def make_logger(work_folder, label, terminal=False):
    loggers = [
        log_utils.CSVLogger(f'./logs/{work_folder}', label=label, add_uid=False)
    ]
    if terminal:
        loggers.append(log_utils.TerminalLogger(label=label))
    
    logger = log_utils.Dispatcher(loggers, log_utils.to_numpy)
    logger = log_utils.NoneFilter(logger)
    # loggers = log_utils.TimeFilter(logger, 1.0)
    return logger

def make_environment(utils, logger = None) -> dm_env.Environment:
    # Make sure the environment obeys the dm_env.Environment interface.
    environment = wrappers.GymWrapper(TradingEnv(
    utils=utils,
    logger=logger))
    # Clip the action returned by the agent to the environment spec.
    # environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)

    return environment

def main(argv):
    gamma_hedge_ratio = 1.0
    if FLAGS.feed_data:
        work_folder = f'greekhedge_spread={FLAGS.spread}_v={FLAGS.vov}_liabttms={FLAGS.liab_ttms}_hedttm={FLAGS.hed_ttm}_hedfrq={FLAGS.hed_frq}_feeddata={FLAGS.feed_data}'
    if FLAGS.feed_data_fx:
        work_folder = f'greekhedge_spread={FLAGS.spread}_v={FLAGS.vov}_liabttms={FLAGS.liab_ttms}_hedttm={FLAGS.hed_ttm}_hedfrq={FLAGS.hed_frq}_feeddatafx={FLAGS.feed_data_fx}'
    if FLAGS.logger_prefix:
        work_folder = FLAGS.logger_prefix + "/" + work_folder
    # Create an environment, grab the spec, and use it to create networks.
    eval_utils = Utils(init_ttm=FLAGS.init_ttm, np_seed=4321, num_sim=FLAGS.eval_sim, spread=FLAGS.spread, volvol=FLAGS.vov, 
                       sabr=FLAGS.sabr, gbm=FLAGS.gbm, hed_ttm=FLAGS.hed_ttm,
                       frq=FLAGS.hed_frq, feed_data=FLAGS.feed_data, fx_frq=FLAGS.hed_frq, feed_data_fx=FLAGS.feed_data_fx,
                       init_vol=FLAGS.init_vol, poisson_rate=FLAGS.poisson_rate, 
                       moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std, 
                       mu=FLAGS.mu, ttms=[int(ttm) for ttm in FLAGS.liab_ttms])
    # Create the evaluation actor and loop.

    if FLAGS.strategy == 'gamma':
        # gamma hedging
        eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, f'eval_gamma_env'))
        eval_actor = GammaHedgeAgent(eval_env, gamma_hedge_ratio)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, f'eval_gamma_loop',True))
        
    elif FLAGS.strategy == 'delta':
        # delta hedging
        eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, 'eval_delta_env'))
        eval_actor = DeltaHedgeAgent(eval_env, gamma_hedge_ratio)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, 'eval_delta_loop', True))
        
    elif FLAGS.strategy == 'vega':
        # vega hedging
        eval_env = make_environment(utils=eval_utils, logger=make_logger(work_folder, 'eval_vega_env'))
        eval_actor = VegaHedgeAgent(eval_env)
        eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=make_logger(work_folder, 'eval_vega_loop', True))

    if FLAGS.feed_data or FLAGS.feed_data_fx:
        eval_loop.run(num_episodes=eval_utils.num_sim)   # the number of paths when feed_data=True
    elif (not FLAGS.feed_data) and (not FLAGS.feed_data_fx):
        eval_loop.run(num_episodes=FLAGS.eval_sim)

    Path(f'./logs/{work_folder}/ok').touch()

if __name__ == '__main__':
    app.run(main)
