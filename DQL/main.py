import tensorflow as tf
import gymnasium as gym
import numpy as np
import random
import statistics
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


D = []
#mbs = []
oi = 0

minibatch_size = 5
memsize = 100 * minibatch_size
n_episodes = 400
n_extra = 5
n_dense = 40
rand_eps = 1
gamma = 0.9

Q = tf.keras.Sequential()
Q.add(tf.keras.layers.Dense(n_dense, input_shape=(3,), activation = 'relu'))
Q.add(tf.keras.layers.Dense(n_dense, activation = 'relu'))
Q.add(tf.keras.layers.Dense(1))

opt = tf.keras.optimizers.SGD(learning_rate=1e-3)

Q.compile(optimizer=opt)

QE = lambda x: Q(tf.constant([x]))

env_map = generate_random_map(size=5)
list(map(print,env_map))

env = gym.make("FrozenLake-v1", desc=env_map, is_slippery = False)

def split(obvs):
    pck = (len(env_map),)
    r = []
    for i in pck:
        r.append(obvs%i)
        obvs //= i
    r.append(obvs)
    return r

for eps in range(n_episodes+n_extra):
    if eps >= n_episodes:
        env = gym.make("FrozenLake-v1", desc=env_map, render_mode="human", is_slippery=False)
    obvs, info = env.reset()

    losses = []
    rewards = []

    for _ in range(200):
        obs = split(obvs)

        if random.random() < max(0.01,rand_eps):
            action = random.randrange(0,4)
        else:
            action = max(range(4), key = lambda x: Q(tf.constant([[x] + obs])))
        

        obvs, reward, trm, truncated, info = env.step(action)
        obs_next = split(obvs)

        rewards.append(reward)

        tr = (obs, action, reward, obs_next, trm)
        if len(D) < memsize:
            D.append(tr)
        else:
            D[oi] = tr
            oi = (oi + 1) % memsize


        minibatch = []
        for (obs, action, reward, obs_next, terminated) in random.sample(D, min((len(D)+1)//2, minibatch_size)):
            y = reward
            if not terminated:
                y += gamma * max(float(Q(tf.constant([[x] + obs_next]))) for x in range(4))
            x = tf.constant([[action] + obs])
            minibatch.append((x,y))

        with tf.GradientTape() as tp:
            loss = 0
            for x,y in minibatch:            
                loss_obs = (y - Q(x))**2
                loss += loss_obs
                losses.append(loss_obs)
        
        grads = tp.gradient(loss, Q.trainable_weights)
        opt.apply_gradients(zip(grads,Q.trainable_weights))
    '''
        loss0 = loss
        loss = 0
        for x,y in minibatch:            
            loss_obs = (y - Q(x))**2
            loss += loss_obs
            losses.append(loss_obs)
        print(loss0 - loss)
    '''

        #mbs.append([(list(x.numpy()),y) for x,y in minibatch])
        
        
        if trm:
            break
    
    print(f"run {eps+1} done: mean loss = {statistics.mean(map(float,losses))}, mean reward = {statistics.mean(rewards)}, length = {len(rewards)}")

    rand_eps -= 2/n_episodes

env.close()