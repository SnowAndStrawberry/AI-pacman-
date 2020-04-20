from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import layout

'''
Team Creation:
(1) Offensive Agent
(2) Defensive Agent
'''


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


'''
Define Reflex Agent extends from capture Agent
'''


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)

        CaptureAgent.registerInitialState(self, gameState)

        self.catchState = False

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


'''
Offensive Agent:
(1) Become pacman to eat opponent food
'''


class OffensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.catchState = False

    def chooseAction(self, gameState):
        # When pacman was eaten by opponent
        opponentCapsule = self.getCapsules(gameState)
        myState = gameState.getAgentState(self.index)

        if gameState.getAgentPosition(self.index) == self.start:
            self.catchState = False

        if self.catchState:
            goalPoint = self.start
        elif myState.numCarrying >= 5 and myState.isPacman:
            goalPoint = self.start
        else:
            goalPoint = None

        action = self.aStarSearch(gameState, goalPoint)
        self.checkCatch(gameState.generateSuccessor(self.index, action))

        return action

    def getDistance(self, gameState, successorPos, goalPoint):
        minDistance = 9999
        if goalPoint is None:
            foodList = self.getFood(gameState).asList()
            capsuleList = self.getCapsules(gameState)
            eatingList = foodList + capsuleList

            if successorPos in eatingList:
                # distance reward is -100
                distance = 0 + (-100)
            else:  # find minimum distance to food
                # distance = min([self.getMazeDistance(successorPos, food) for food in foodList])
                for food in eatingList:
                    if self.getMazeDistance(successorPos, food) < minDistance:
                        minDistance = self.getMazeDistance(successorPos, food)
                distance = minDistance

        else:
            # distance to start point
            distance = self.getMazeDistance(successorPos, self.start)

            if self.getMazeDistance(gameState.getAgentPosition(self.index), self.start) >= 2:
                if successorPos == self.start:
                    distance = 9999999  # being catch and go home, so this distance should be the biggest one
        return distance

    def checkCatch(self, successor):
        position = successor.getAgentPosition(self.index)
        myState = successor.getAgentState(self.index)

        minDistance = 9999
        opponentGhost = []

        opponentsIndices = self.getOpponents(successor)
        for opponentIndex in opponentsIndices:
            opponent = successor.getAgentState(opponentIndex)

            if not opponent.isPacman and opponent.getPosition() is not None:
                oppentPos = opponent.getPosition()
                disToOppent = self.getMazeDistance(position, oppentPos)
                if disToOppent < minDistance:
                    minDistance = disToOppent
                    opponentGhost.append(opponent)

        if len(opponentGhost) > 0 and minDistance <= 3:
            # ghost scared:
            if opponentGhost[-1].scaredTimer > 0 and (myState.numCarrying <= 5):
                self.catchState = False

            else:
                self.catchState = True
        else:
            self.catchState = False

    def aStarSearch(self, gameState, goal):

        # Initialization
        explored = []
        exploring = util.PriorityQueue()
        exploring.push([gameState, []], 0)
        loop = False
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        eatingList = foodList + capsuleList

        # Begin search
        while not loop:
            popItem = exploring.pop()
            currentState = popItem[0]
            beforeAction = popItem[1]
            currentPos = currentState.getAgentPosition(self.index)

            # Find goal point
            if currentPos == goal or (currentPos in eatingList and not self.catchState):
                loop = True
                return beforeAction[0]

            if currentPos in explored:
                continue
            else:
                explored.append(currentPos)
                legalAction = currentState.getLegalActions(self.index)

            for action in legalAction:
                successor = currentState.generateSuccessor(self.index, action)
                successorPos = successor.getAgentPosition(self.index)
                    # ghost Pos
                enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
                ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]

                if len(ghosts) > 0:
                    dists = [self.getMazeDistance(successorPos, a.getPosition()) for a in ghosts]

                    if min(dists) < 5:
                        fx = self.getDistance(currentState, successorPos, goal) + (-100) * min(dists)
                    else:
                        fx = self.getDistance(currentState, successorPos, goal) + (-100) * 10
                else:
                    fx = self.getDistance(currentState, successorPos, goal) + (-100) * 10

                item = [successor, beforeAction + [action]]
                exploring.push(item, fx)


'''
Defensive Agent
(1) Keep walking in the center of the map
(2) Will track opponent position
'''


class DefensiveAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # Get distance
        self.distancer.getMazeDistances()
        # Set defensive area
        self.setDefensiveArea(gameState)
        self.start = gameState.getAgentPosition(self.index)

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.catchState = False
        self.coreDefendingArea = []
        self.target = None
        self.remainFoodList = []
        self.number = 0
        self.catchState = False

    def getMapInfo(self, gameState):
        layoutInfo = []
        layoutWidth = gameState.data.layout.width
        layoutHeight = gameState.data.layout.height
        layoutCentralX = (layoutWidth - 2) // 2
        if not self.red:
            layoutCentralX += 1
        layoutCentralY = (layoutHeight - 2) // 2
        layoutInfo.extend((layoutWidth, layoutHeight, layoutCentralX, layoutCentralY))
        return layoutInfo

    def setDefensiveArea(self, gameState):

        layoutInfo = self.getMapInfo(gameState)

        for i in range(1, layoutInfo[1] - 1):
            if not gameState.hasWall(layoutInfo[2], i):
                self.coreDefendingArea.append((layoutInfo[2], i))

        desiredSize = layoutInfo[3]
        currentSize = len(self.coreDefendingArea)

        while desiredSize < currentSize:
            self.coreDefendingArea.remove(self.coreDefendingArea[0])
            self.coreDefendingArea.remove(self.coreDefendingArea[-1])
            currentSize = len(self.coreDefendingArea)

        while len(self.coreDefendingArea) > 2:
            self.coreDefendingArea.remove(self.coreDefendingArea[0])
            self.coreDefendingArea.remove(self.coreDefendingArea[-1])

    def chooseAction(self, gameState):
        # Our home food list
        ourCurrentFoodList = self.getFoodYouAreDefending(gameState).asList()
        # our position
        myPos = gameState.getAgentPosition(self.index)
        if myPos == self.target:
            self.target = None
        # Get the cloest invader's position and set target as invader
        opponentsInfo = []
        normalOpponPacmanPos = []
        nearestPacman = []
        # set distance to infinity
        minDistance = 99999

        myScore = self.getScore(gameState)
        myState = gameState.getAgentState(self.index)

        opponentsInfo = self.getOpponents(gameState)
        for opponentIndex in opponentsInfo:
            opponent = gameState.getAgentState(opponentIndex)
            # opponent are eating our food
            if opponent.isPacman and opponent.getPosition() is not None:
                # opponent pacman position
                opponentPos = opponent.getPosition()
                normalOpponPacmanPos.append(opponentPos)

        # When opponent pacman can be eaten by us
        if len(normalOpponPacmanPos) > 0:
            for position in normalOpponPacmanPos:
                # find the nearest opponent pacman
                distance = self.getMazeDistance(position, myPos)
                if distance < minDistance:
                    minDistance = distance
                    nearestPacman.append(position)

            self.target = nearestPacman[-1]
        # get the eaten food position
        else:
            if len(self.remainFoodList) > 0 and len(ourCurrentFoodList) < len(self.remainFoodList):
                eatenFood = set(self.remainFoodList) - set(ourCurrentFoodList)

                self.target = eatenFood.pop()

        self.remainFoodList = ourCurrentFoodList

        if self.target is None:
            if len(ourCurrentFoodList) <= 4:
                highPriorityFood = ourCurrentFoodList + self.getCapsulesYouAreDefending(gameState)
                self.target = random.choice(highPriorityFood)
            else:
                self.target = random.choice(self.coreDefendingArea)
        # evaluates candiateActions and get the best
        candidateActions = self.defensiveStart(gameState)
        goodActions = []
        fValues = []

        for a in candidateActions:
            new_state = gameState.generateSuccessor(self.index, a)
            newPos = new_state.getAgentPosition(self.index)
            goodActions.append(a)
            fValues.append(self.getMazeDistance(newPos, self.target))

        best = min(fValues)
        bestActions = [a for a, v in zip(goodActions, fValues) if v == best]
        bestAction = random.choice(bestActions)

        return bestAction

    def defensiveStart(self, gameState):
        candidateActions = []
        actions = gameState.getLegalActions(self.index)
        reversed_direction = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        actions.remove(Directions.STOP)
        if reversed_direction in actions:
            actions.remove(reversed_direction)

        for action in actions:
            newState = gameState.generateSuccessor(self.index, action)
            if not newState.getAgentState(self.index).isPacman:
                candidateActions.append(action)

        if len(candidateActions) == 0:
            self.number = 0
        else:
            self.number = self.number + 1

        if self.number > 20 or self.number == 0:
            candidateActions.append(reversed_direction)

        return candidateActions
