from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions,Actions
import game
from util import nearestPoint
import layout
import numpy as np

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


class OffensiveAgent(ReflexCaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.catchState = False
        self.epsilon = 0.0
        self.alpha = 0.1
        self.discountRate = 0.9

    def chooseAction(self, gameState):

        opponentCapsule = self.getCapsules(gameState)
        myState = gameState.getAgentState(self.index)
        legalActions = gameState.getLegalActions(self.index)

        if gameState.getAgentPosition(self.index) == self.start:
            self.catchState = False

        if self.catchState:
            goalPoint = self.start
            action = self.aStarSearch(gameState, goalPoint)
        elif myState.numCarrying >= 5 and myState.isPacman:
            goalPoint = self.start
            action = self.aStarSearch(gameState, goalPoint)
        else:
            flag = util.flipCoin(self.epsilon)
            if flag:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(gameState)
        self.checkCatch(gameState.generateSuccessor(self.index, action))

        return action

    def getPolicy(self, gameState):
        values = []
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        if len(legalActions) == 0:
            return None
        else:
            for action in legalActions:
                # self.updateWeights(gameState, action)
                values.append((self.evaluate(gameState, action), action))
        return max(values)[1]

    def getValue(self, gameState):
        qValues = []
        legalActions = gameState.getLegalActions(self.index)
        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                qValues.append(self.evaluate(gameState, action))
            return max(qValues)

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        food = self.getFood(gameState)
        capsule = self.getCapsules(gameState)
        walls = gameState.getWalls()
        next_x, next_y = self.getNextpos(gameState, action)

        features['successorScore'] = self.getScore(successor)

        features["bias"] = 1.0

        if food[next_x][next_y]:
            features["eatFood"] = 1.0

        #if capsule[next_x][next_y]:
            #features["eatFood"] = 1.0

        closefood = self.closeFood((next_x, next_y), food, walls)
        if closefood is not None:
            features["closeFood"] = float(closefood) / (walls.width * walls.height)

        return features

    def closeFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, closefood = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            if food[pos_x][pos_y]:
                return closefood
            lns = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for lns_x, lns_y in lns:
                fringe.append((lns_x, lns_y, closefood + 1))
        return None

    def getNextpos(self,gameState, action):
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        return next_x, next_y

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        """
        try:
            with open('weights.txt', "r") as file:
                self.weights = eval(file.read())
                return self.weight
        except IOError:
                return
        """
        return {'closeFood': -2.2612110712568386, 'bias': 1.1063213368231456, 'successorScore': -0.029456432758495672,
                'eatFood': 9.982134634265219}


    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)
        reward = nextState.getScore() - gameState.getScore()

        for feature in features:
            correction = (reward + self.discountRate * self.getValue(nextState)) - self.evaluate(gameState, action)
            self.weights[feature] = self.weights[feature] + self.alpha * correction * features[feature]


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

            if self.getMazeDistance(gameState.getAgentPosition(self.index), self.start) > 1:
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
            if opponentGhost[-1].scaredTimer > 0:
                self.catchState = False
            else:
                self.catchState = True
        else:
            self.catchState = False

    def aStarSearch(self, gameState, goalPoint):

        ############### initialization ##################
        explored = []
        exploring = util.PriorityQueue()
        legalAction = []
        done = False
        exploring.push([gameState, []], 0)
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        eatingList = foodList + capsuleList
        #################################################
        while not done:
            popItem = exploring.pop()
            currentState = popItem[0]
            beforeAction = popItem[1]
            currentPos = currentState.getAgentPosition(self.index)

            ## when find the needed point ##
            if currentPos == goalPoint or (currentPos in eatingList and not self.catchState):
                done = True
                return beforeAction[0]

            ## avoid duplicate exploration ##
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

                    if min(dists) < 4:
                        fx = self.getDistance(currentState, successorPos, goalPoint) + (-100) * min(dists)
                    else:
                        fx = self.getDistance(currentState, successorPos, goalPoint) + (-100) * 10
                else:
                    fx = self.getDistance(currentState, successorPos, goalPoint) + (-100) * 10

                ## the item pushed into priorityQueue
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
        self.isFoodEaten = False
        self.patrolDict = {}
        self.tick = 0
        self.gazeboDict = {}
        self.catchState = False

    def chooseAction(self, gameState):
        # Our home food list
        ourCurrentFoodList = self.getFoodYouAreDefending(gameState).asList()
        # our position
        myPos = gameState.getAgentPosition(self.index)
        if myPos == self.target:
            self.target = None
        # Get the cloest invader's position and set target as invader
        opponentsIndices = []
        threateningInvaderPos = []
        cloestInvaders = []
        # set distance to infinity
        minDistance = 99999

        myScore = self.getScore(gameState)
        myState = gameState.getAgentState(self.index)

        opponentsIndices = self.getOpponents(gameState)
        for opponentIndex in opponentsIndices:
            opponent = gameState.getAgentState(opponentIndex)
            # opponent are eating our food
            if opponent.isPacman and opponent.getPosition() is not None:
                opponentPos = opponent.getPosition()
                threateningInvaderPos.append(opponentPos)

        # When opponent pacman can be eaten by us
        if len(threateningInvaderPos) > 0:
            for position in threateningInvaderPos:
                distance = self.getMazeDistance(position, myPos)
                if distance < minDistance:
                    minDistance = distance
                    cloestInvaders.append(position)
            self.target = cloestInvaders[-1]

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
        candidateActions = self.ForcedDefend(gameState)
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

    def ForcedDefend(self, gameState):
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
            self.tick = 0
        else:
            self.tick = self.tick + 1

        if self.tick > 20 or self.tick == 0:
            candidateActions.append(reversed_direction)

        return candidateActions


# def final(self, gameState):
# print self.weights
# file = open('weights.txt', 'w')
# file.write(str(self.weights))