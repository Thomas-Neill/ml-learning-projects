import tensorflow as tf
import gymnasium as gym
import numpy as np
import random
import statistics

D = []
oi = 0

memsize = 100
minibatch_size = 32
n_episodes = 1000
n_dense = 16
rand_eps = 1
gamma = 0.9

Q = tf.keras.Sequential()
Q.add(tf.keras.layers.Dense(n_dense, input_shape=(3,), activation = 'relu'))
Q.add(tf.keras.layers.Dense(n_dense, activation = 'relu'))
Q.add(tf.keras.layers.Dense(1))

opt = tf.keras.optimizers.SGD(learning_rate=1e-4)

Q.compile(optimizer=opt)

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4")

def split(obvs):
    pck = (4,)
    r = []
    for i in pck:
        r.append(obvs%i)
        obvs //= i
    r.append(obvs)
    return r

for eps in range(n_episodes+10):
    if eps >= n_episodes:
        env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode="human")
        input("done!")
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

        for (obs, action, reward, obs_next, terminated) in random.sample(D, min((len(D)+1)//2, minibatch_size)):
            y = reward
            if not terminated:
                y += gamma * max(Q(tf.constant([[x] + obs_next])) for x in range(4))

            with tf.GradientTape() as tp:
                loss = (y - Q(tf.constant([[action] + obs])))**2
            losses.append(loss)
            grads = tp.gradient(loss, Q.trainable_weights)
            opt.apply_gradients(zip(grads,Q.trainable_weights))
        
        if trm:
            break
    
    print(f"run {eps+1} done: mean loss = {statistics.mean(map(float,losses))}, mean reward = {statistics.mean(rewards)}, length = {len(rewards)}")

    rand_eps -= 1/600



env.close()

while True:
    try:
        eval(input())
    except:
        pass