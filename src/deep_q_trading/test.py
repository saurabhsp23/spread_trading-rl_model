def test(agent, test_env, output_path):
    state = test_env.reset()
    actions = []
    while True:
        action = agent.exploit(state)
        next_state, reward, done = test_env.step(action)
        actions.append(action)

        if done:
            save_test_results(output_path, test_env, actions)
            break

        state = next_state