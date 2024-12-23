import os

def train(agent, train_env, batch_size, num_episodes, output_path):
    for episode in range(num_episodes):
        print(f"Training Episode {episode}")
        state = train_env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = train_env.step(action)
            agent.memory.push(state, action, reward, next_state, done)

            if train_env.current_step % 100 == 0:
                agent.update(batch_size)

            if done:
                save_training_results(output_path, episode, train_env)
                break

            state = next_state

        # Save model periodically
        if episode % 5 == 0:
            torch.save(agent.model.state_dict(), f"model-{episode}.pth")
