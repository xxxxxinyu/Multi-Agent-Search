from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)

        def MiniMax(gameState, agentIndex, depth=0):
            # Get all legal actions 
            legalActionList = gameState.getLegalActions(agentIndex)
            numIndex = gameState.getNumAgents() - 1 
            bestAction = None

            # If node is terminal, then return score
            if (gameState.isLose() or gameState.isWin() or (depth == self.depth)):
                return [self.evaluationFunction(gameState)]
            elif agentIndex == numIndex: # Pacman and Ghost's turns are over -> next depth
                depth += 1
                childAgentIndex = self.index
            else: # Increase agent_index by 1 as it will be next player's turn now
                childAgentIndex = agentIndex + 1
            # Ghost
            if agentIndex != 0:
                v = float("inf") # Initialize
                for legalAction in legalActionList: # For each legal action of ghost agent
                    # Generate successor
                    successorGameState = gameState.getNextState(agentIndex, legalAction)
                    # Get the minimax score of successor
                    min = MiniMax(successorGameState, childAgentIndex, depth)[0]
                    if min == v:
                        if bool(random.getrandbits(1)):
                            bestAction = legalAction # choose bestaction randomly
                    elif min < v:
                        v = min
                        bestAction = legalAction
                return v, bestAction
            # Pacman
            else:
                v = -float("inf") # Initialize
                for legalAction in legalActionList: # For each legal action of Pacman
                    # Generate successor
                    successorGameState = gameState.getNextState(agentIndex, legalAction)
                    # Get the minimax score of successor
                    max = MiniMax(successorGameState, childAgentIndex, depth)[0]
                    if max == v:
                        if bool(random.getrandbits(1)):
                            bestAction = legalAction # choose bestaction randomly
                    elif max > v:
                        v = max
                        bestAction = legalAction
                return v, bestAction

        bestScoreActionPair = MiniMax(gameState, self.index)
        bestScore = bestScoreActionPair[0]
        bestMove =  bestScoreActionPair[1]

        return bestMove

        raise NotImplementedError("To be implemented")
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)

        def max_agent(state, depth, alpha, beta):
            # If node is terminal, then return score
            if state.isWin() or state.isLose():
                return state.getScore()
            # Get all legal actions of Pacman
            legalActionList = state.getLegalActions(0) 
            # Initalize
            bestScore = float("-inf")
            score = bestScore
            bestAction = Directions.STOP

            for legalAction in legalActionList: # For each legal action of Pacman
                # For each successor of state: score = max(score, value(successor, α, β))
                score = min_agent(state.getNextState(0, legalAction), depth, 1, alpha, beta)
                if score > bestScore:
                    bestScore = score
                    bestAction = legalAction
                # Updata the value of alpha
                alpha = max(alpha, bestScore)
                # Prune 
                if bestScore > beta:
                    return bestScore
            if depth == 0:
                return bestAction
            else:
                return bestScore

        def min_agent(state, depth, ghost, alpha, beta):
            # If node is terminal, then return score
            if state.isLose() or state.isWin():
                return state.getScore()
            next_player = ghost + 1 
            if ghost == state.getNumAgents() - 1: # All ghost are over
                next_player = 0
            # Get all legal actions of Ghost
            legalActionList = state.getLegalActions(ghost)
            # Initalize
            bestScore = float("inf")
            score = bestScore
            for legalAction in legalActionList: # For each legal action of Ghost
                if next_player == 0: # On the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1: # If node is terminal, return the evaluation of score
                        score = self.evaluationFunction(state.getNextState(ghost, legalAction))
                    else: # If not, call max_agent
                        score = max_agent(state.getNextState(ghost, legalAction), depth + 1, alpha, beta)
                else: # For Ghost
                    # For each successor of state: score = min(score, value(successor, α, β))
                    score = min_agent(state.getNextState(ghost, legalAction), depth, next_player, alpha, beta)
                if score < bestScore:
                    bestScore = score
                # Update the value of beta
                beta = min(beta, bestScore)
                # Prune
                if bestScore < alpha:
                    return bestScore
            return bestScore
        return max_agent(gameState, 0, float("-inf"), float("inf"))
        raise NotImplementedError("To be implemented")
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        def expectimax(gameState, agentIndex, depth=0):
            # Get all legal actions 
            legalActionList = gameState.getLegalActions(agentIndex)
            numIndex = gameState.getNumAgents() - 1
            bestAction = None
            # If node is terminal, then return score
            if (gameState.isLose() or gameState.isWin() or depth == self.depth): # Pacman and Ghost's turns are over -> next depth
                return [self.evaluationFunction(gameState)]
            elif agentIndex == numIndex:
                depth += 1
                childAgentIndex = self.index
            else:
                childAgentIndex = agentIndex + 1

            numAction = len(legalActionList)
            # If player(pos) == MAX: value = -infinity
            if agentIndex == self.index:
                value = -float("inf")
            # If player(pos) == CHANCE: value = 0
            else:
                value = 0

            for legalAction in legalActionList: # For each legal action of player
                # Generate successor
                successorGameState = gameState.getNextState(agentIndex, legalAction)
                expectedMax = expectimax(successorGameState, childAgentIndex, depth)[0]
                if agentIndex == self.index:
                    if expectedMax > value:
                        # value, best_move = nxt_val, move
                        value = expectedMax
                        bestAction = legalAction
                else:
                    # value = value + prob(move) * nxt_val
                    value = value + ((1.0/numAction) * expectedMax)
            return value, bestAction

        bestScoreActionPair = expectimax(gameState, self.index)
        bestScore = bestScoreActionPair[0]
        bestMove =  bestScoreActionPair[1]
        return bestMove
        raise NotImplementedError("To be implemented")
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """
    States are evaluated into sections that linearly correspond to the total 
    of the evaluation: win, lose, score, foodScore, and ghost.
    Best case scenario is win, so win contributes the most to an evaluationFunction.
    Score is second important, so it scales up by 10K.
    In a state, we want to get rid of food, capsules, and ghosts (when they
    are to be eaten) so as Pacman gets closer, their respective evaluation
    scales down.
    Avoidance of ghosts is important, but not that important. Ghosts do not
    run faster than Pacman, and are only a real threat if they are 1 step away.
    A light heuristic to get far away from the ghosts as necessary (without
    being too scared from getting food) is all that is needed.

    """
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    numCapsules = len(currentGameState.getCapsules())
    foodList = currentGameState.getFood().asList()
    numFood = currentGameState.getNumFood()
    # Initialize
    badGhost = []
    yummyGhost = []
    total = 0
    win = 0
    lose = 0
    score = 0
    foodScore = 0
    ghost = 0
    if currentGameState.isWin():
        win = 10000000000000000000000000000
    elif currentGameState.isLose():
        lose = -10000000000000000000000000000
    score = 10000 * currentGameState.getScore()
    capsules = 10000000000/(numCapsules+1)
    for food in foodList:
        foodScore += 50/(manhattanDistance(pacmanPosition, food)) * numFood
    for index in range(len(scaredTimes)):
        if scaredTimes[index] == 0:
            badGhost.append(ghostPositions[index])
        else:
            yummyGhost.append(ghostPositions[index])
    for index in range(len(yummyGhost)):
        ghost += 1/(((manhattanDistance(pacmanPosition, yummyGhost[index])) * scaredTimes[index])+1)
    for death in badGhost:
        ghost +=  manhattanDistance(pacmanPosition, death)
    total = win + lose + score + capsules + foodScore + ghost
    return total
    raise NotImplementedError("To be implemented")
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
