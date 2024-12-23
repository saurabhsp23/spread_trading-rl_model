def validate(agent, valid_env, output_path, best_valid_reward, patience):
    state = valid_env.reset()
    no_improve_counter = 0

    while True:
        action = agent.exploit(state)
        next_state, reward, done = valid_env.step(action)

        if done:
            save_validation_results(output_path, valid_env)
            if valid_env.total_reward > best_valid_reward:
                best_valid_reward = valid_env.total_reward
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter > patience:
                print("Early stopping due to lack of improvement.")
                break

            return best_valid_reward, no_improve_counter

        state = next_state
