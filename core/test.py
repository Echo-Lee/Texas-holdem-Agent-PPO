import numpy as np
from pettingzoo.classic import leduc_holdem_v4

def run_test(use_break: bool):
    env = leduc_holdem_v4.env()
    env.reset(seed=42) # 固定随机种子，保证两局游戏的走法完全一模一样
    
    print(f"\n{'='*40}")
    print(f"🚀 开始测试: 使用 {'[ break ]' if use_break else '[ continue ]'} 处理结束信号")
    print(f"{'='*40}")

    # 用于记录玩家在这一局里收到的所有 reward
    rewards_record = {"player_0": [], "player_1": []}

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, _ = env.last()
        done = termination or truncation

        if done:
            print(f"💀 [{agent_name}] 收到游戏结束信号(done=True)！它的最终 Reward 是: {reward}")
            rewards_record[agent_name].append(reward)
            
            env.step(None) # 踩一脚 None，告诉环境我已经收到死亡通知单了
            
            if use_break:
                print(f"   ❌ 触发 break！直接暴力砸碎了整个 for 循环。")
                break
            else:
                print(f"   ✅ 触发 continue！让环境去通知下一个玩家。")
                continue

        # 如果游戏没结束，随机选一个合法动作
        valid_actions = np.where(observation["action_mask"] == 1)[0]
        action = np.random.choice(valid_actions)
        
        print(f"➡️  [{agent_name}] 游戏进行中... 收到上一步 Reward: {reward}")
        rewards_record[agent_name].append(reward)
        
        env.step(action)

    print("\n📊 --- 游戏结束，奖励结算核对 ---")
    print(f"Player 0 记录到的所有奖励: {rewards_record['player_0']}")
    print(f"Player 1 记录到的所有奖励: {rewards_record['player_1']}")
    
    # 验证零和博弈性质
    p0_total = sum(rewards_record['player_0'])
    p1_total = sum(rewards_record['player_1'])
    print(f"💰 最终筹码变动 -> P0: {p0_total}, P1: {p1_total}")
    if p0_total + p1_total != 0:
        print("🚨 警告：这不符合零和博弈！有人输的钱凭空消失了！")
    else:
        print("🎉 完美：输赢总和为 0，数据记录完整。")


if __name__ == "__main__":
    # 1. 运行你的旧逻辑 (使用 break)
    run_test(use_break=True)
    
    # 2. 运行修复后的逻辑 (使用 continue)
    run_test(use_break=False)