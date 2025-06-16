# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    """
    I changed the noise to 0.0 because, with that the agent can easily cross
    the bridge without any fear of falling. 
    The discount is set to 0.9 because it will allow the agent to value the 
    future rewards enough to make crossing the bridge worth it.
    Wiht no noise, the path is determinitic, and the agent will always choose 
    reach the hgih reward state at the end
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """
    """
    discount of 0.9 makes the agent value future rewards, but that much
    so that it won't travel all the way to the far exit
    zero noise makes the agent take the direct path, even if its risky, 
    which it would be.
    -5 was not an exact number to be choosen, it is just a very low reward,
    a negative reward, this is just make it like urgent to reach the chose exit
    quickly. 
    
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = -5.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """

    """
    discount of 0.3 is low and makes the agent foucs on the the rewards easist 
    to get, adn less on the future rewards, which will encourge it to take the
    closer exit of 1.0. 
    noise of 0.2 makes the agent cautious about the cliff, so it wont get too close
    to it. 
    0 living reward is just a number, that wouldnt make it urgent, or slow it down to
    reach the exit, this way, it can take the path thats safer even if its longer. 
    """
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """
    """
    high discount of 0.9 makes the agent value future rewards, as discussed before
    0 noise makes it so the agent can take the direct path even if its risky (as before)
    the reward is very small for a negatvie number, so that it create urgenct, but not so much
    that it ends up going to the close exit.
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = -0.5
    
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    """
    high discount of 0.9 makes the agent value future rewards, as discussed before
    0.2 noise makes the agent cautious about the risky path, as discussed before
    this is again a very small negative reward, which creates some urgency, but lets the
    agent take a longer path if its safer. 
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """
    """
    as before, high discount will make the agent value future rewards
    as before, 0.2 noise will make the agent cautious about the cliff
    a very very high living reward is not required, it can be smaller, 
    however, the high living reward will make the agent value staying alive, 
    the terminating,
    13 is higher than any exit reward possible, so the agents just keeps moving 
    the grid. 
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 13.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question7():
  """
  with high epsilon, the agents explores too much and doesn't even exploit 
  the optiomal policy enough withing the 50 episodes
  with low epsilon, the agent probably wont even explore enough to discover
  the optimal path across the bridge. 
  with high learning rate, the agent is just gonna update too drastically 
  based on single experiences, and that will lead to unstable learning
  with low learning rate, the agent is gonna learn too slowly to converge 
  in 50 episodes. 
  
  so because a balance can't be made with >99% accuracy, the answer is not possible
  """
  return "NOT POSSIBLE"

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
