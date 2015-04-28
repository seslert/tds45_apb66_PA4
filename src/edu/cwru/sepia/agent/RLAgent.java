package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.util.DistanceMetrics;

import java.io.*;
import java.util.*;

/**
 * This class defines an agent to control footmen in the given map configurations.
 * @course EECS 391: Introduction to Artificial Intelligence
 * @project PA4
 * @author Timothy Sesler
 * @author Adam Boe
 * @date 23 April 2015
 *
 */
public class RLAgent extends Agent {

    /**
     * Set in the constructor. Defines how many learning episodes your agent should run for.
     * When starting an episode. If the count is greater than this value print a message
     * and call sys.exit(0)
     */
    public final int numEpisodes;
    public int totalCompletedEpisodes = 0;
    public int completedLearningEpisodes = 0;
    public int completedTestingEpisodes = 0;
    
    public Map<Integer, Double> footmenRewardMap;	// Maps myFootmen ID to their total reward
    
    public Double[] testingRewards;
    public double cumulativeReward;
    public List<Double> averageCumulativeRewards;

    private List<Integer> myFootmen;	// Your footmen
    private List<Integer> enemyFootmen;	// Enemy's footmen
    private boolean freezeForEvaluation = false;
    private double maxQValue;

    /**
     * Convenience variable specifying enemy agent number. Use this whenever referring
     * to the enemy agent. We will make sure it is set to the proper number when testing your code.
     */
    public static final int ENEMY_PLAYERNUM = 1;

    /**
     * Set this to whatever size your feature vector is.
     */
    public static final int NUM_FEATURES = 5;

    /** Use this random number generator for your epsilon exploration. When you submit we will
     * change this seed so make sure that your agent works for more than the default seed.
     */
    public final Random random = new Random(12345);

    public Double[] weights;	// Your Q-function weights.
    public Double[] featureVector;	// Old feature vector.

    /**
     * These variables are set for you according to the assignment definition. You can change them,
     * but it is not recommended. If you do change them please let us know and explain your reasoning for
     * changing them.
     */
    public final double gamma = 0.9;
    public final double learningRate = .0001;
    public final double epsilon = .02;

    public RLAgent(int playernum, String[] args) {
        super(playernum);

        if (args.length >= 1) {
            numEpisodes = Integer.parseInt(args[0]);
            System.out.println("Running " + numEpisodes + " episodes.");
        } 
        else {
            numEpisodes = 10;
            System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
        }

        boolean loadWeights = false;
        
        if (args.length >= 2) {
            loadWeights = Boolean.parseBoolean(args[1]);
        } 
        else {
            System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
        }

        if (loadWeights) {
            weights = loadWeights();
        }
        else {
            // initialize weights to random values between -1 and 1
            weights = new Double[NUM_FEATURES];
            
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random.nextDouble() * 2 - 1;
            }
        }
        this.cumulativeReward = 0.0;
        this.averageCumulativeRewards = new LinkedList<Double>();
        this.averageCumulativeRewards.add(0.0);
    }

    /**
     * We've implemented some setup code for your convenience. Change what you need to.
     */
    @Override
    public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {
    	
    	this.maxQValue = 0.0;    	

        // You will need to add code to check if you are in a testing or learning episode

        // Find all of your units.
        myFootmen = new LinkedList<>();
        
        for (Integer unitId : stateView.getUnitIds(playernum)) {
            Unit.UnitView unit = stateView.getUnit(unitId);
            String unitName = unit.getTemplateView().getName().toLowerCase();
            
            if (unitName.equals("footman")) {
                myFootmen.add(unitId);
            } 
            else {
                System.err.println("Unknown player unit type: " + unitName);
            }
        }
    
        // Initialize testing reward tracking.
        this.testingRewards = new Double[myFootmen.size()];        

        // Find all of the enemy units.
        enemyFootmen = new LinkedList<>();
        
        for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
            Unit.UnitView unit = stateView.getUnit(unitId);
            String unitName = unit.getTemplateView().getName().toLowerCase();
            
            if (unitName.equals("footman")) {
                enemyFootmen.add(unitId);
            } 
            else {
                System.err.println("Unknown enemy unit type: " + unitName);
            }
        }
        
        // Initialize all footmen with 0 initial reward.
        footmenRewardMap = new HashMap<Integer, Double>();
        
        for (Integer id : myFootmen) {
        	footmenRewardMap.put(id, 0.0);
        }

        return middleStep(stateView, historyView);
    }

    /**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an event whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {
    	
    	Map<Integer, Action> actionMap = new HashMap<Integer, Action>();
    	calculateFootmenRewards(stateView, historyView);	// Update the rewards for the new state.
    	
    	if (significantEvent(stateView, historyView)) {    		    		
    		// Update the weights for each footman.
    		for (Integer id : myFootmen) {
    			int enemyId = selectAction(stateView, historyView, id);
    			
    			if (!freezeForEvaluation) {
        			updateWeights(	this.weights, 
							calculateFeatureVector(stateView, historyView, id, enemyId), 
							footmenRewardMap.get(id), 
							stateView, 
							historyView, 
							id);
    			}
    			// Issue new actions.
    	    	actionMap.put(id, Action.createCompoundAttack(id, enemyId));
    		}
    	}
    	// It's not the first turn.
    	if (stateView.getTurnNumber() > 0) {
    		removeDeadUnits(stateView, historyView);	// Remove any units that were killed in the last turn.
        	Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
        	
    		for (ActionResult ar : actionResults.values()) {
    			//System.out.println(ar.toString());
    		}
    	}    	
    	
        return actionMap;
    }

    /**
     * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
     * finished a set of test episodes you will call out testEpisode.
     *
     * It is also a good idea to save your weights with the saveWeights function.
     */
    @Override
    public void terminalStep(State.StateView stateView, History.HistoryView historyView) {
    	
    	// Last step updates and cleanup.
    	calculateFootmenRewards(stateView, historyView);
    	removeDeadUnits(stateView, historyView);
    	
//    	System.out.println("Player " + playernum + " Units remaining: " + myFootmen.size());
//    	System.out.println("Enemy " + ENEMY_PLAYERNUM + " Units remaining: " + enemyFootmen.size());
    	
    	if (enemyFootmen.size() == 0 && myFootmen.size() > 0) {
    		// We win!
//    		System.out.println("The Player has WON with " + myFootmen.size() + " allied footmen remaining!");
    	}
    	else if (myFootmen.size() == 0 && enemyFootmen.size() > 0) {
    		// We lose!
//    		System.out.println("The Player has LOST with " + enemyFootmen.size() + " enemy footmen remaining!");
    	}
    	
    	// Increment completed learning episodes.
    	if (!freezeForEvaluation && completedLearningEpisodes < 10) {
    		totalCompletedEpisodes++;
    		completedLearningEpisodes++;
    	}
    	// Switch to testing if we've learned for 10 episodes.
    	else if (!freezeForEvaluation) {
    		freezeForEvaluation = true;
    		completedLearningEpisodes = 0;
    	}
    	// Increment completed testing episode.
    	if (freezeForEvaluation && completedTestingEpisodes < 5) {
    		completedTestingEpisodes++;    		
    	}
    	// Switch to learning if we've tested for 5 episodes.
    	else if (freezeForEvaluation) {
    		freezeForEvaluation = false;
    		completedTestingEpisodes = 0;
    		
//    		double sum = 0.0;
    		
//    		for (int i = 0; i < testingRewards.length; i++) {
//    			System.out.println("Reward at " + i + ": " + testingRewards[i]);
//    			
//    			sum += testingRewards[i];
//    		}
    		System.out.println("Average reward for this session: " + (cumulativeReward / 5));
    		averageCumulativeRewards.add(cumulativeReward / 5);
    		cumulativeReward = 0.0;
    	}
    	
    	// Increment total completed episodes.
//    	totalCompletedEpisodes++;
//    	System.out.println("Total completed episodes: " + totalCompletedEpisodes);    
//    	System.out.println("Learning episodes: " + completedLearningEpisodes);
//    	System.out.println("Testing episodes: " + completedTestingEpisodes);

    	if (totalCompletedEpisodes > numEpisodes) {
    		System.out.println("Complete.");
    		printTestData(averageCumulativeRewards);
    		System.exit(0);
    	}

        // MAKE SURE YOU CALL printTestData after you finish a test episode.

        // Save your weights
        saveWeights(weights);
    }

    /**
     * Calculate the updated weights for this agent. 
     * @param previousWeights Weights prior to update
     * @param previousFeatureVector Features from (s,a)
     * @param totalReward Cumulative discounted reward for this footman.
     * @param stateView Current state of the game.
     * @param historyView History of the game up until this point
     * @param footmanId The footman we are updating the weights for
     * @return The updated weight vector.
     */
    public double[] updateWeights(Double[] previousWeights, double[] previousFeatureVector, Double totalReward, State.StateView stateView, History.HistoryView historyView, int footmanId) {
    	
    	double[] updatedWeights = new double[previousWeights.length];
    	
    	// wi = wi - [learningrate(-R(s, a) + gamma * max a'[q'(s,a) - q(s,a)])f(s, a)]
    	for (int i = 0; i < previousWeights.length; i++) {
    		// Modify updatedWeights[i]
    		double currentQValue = 0.0;
    		
    		for (int j = 0; j < previousFeatureVector.length; j++) {
    			currentQValue += previousWeights[j] * previousFeatureVector[j];
    		}
    		double maxQValue = Double.NEGATIVE_INFINITY;
    		
    		for (Integer enemyId : enemyFootmen) {
    			double tempQValue = calcQValue(stateView, historyView, footmanId, enemyId);
    			
    			if (tempQValue > maxQValue) {
    				maxQValue = tempQValue;
    			}
    		}
    		
    		double targetQValue = totalReward + gamma * maxQValue;
    		double squaredLoss = -1 * (targetQValue - currentQValue) * previousFeatureVector[i];    		
    		
    		updatedWeights[i] = previousWeights[i] - learningRate * squaredLoss;    		
    	}
    	
    	return updatedWeights;
    }

    /**
     * Given a footman and the current state and history of the game select the enemy that 
     * this unit should attack. This is where you would do the epsilon-greedy action selection.
     *
     * @param stateView Current state of the game
     * @param historyView The entire history of this episode
     * @param attackerId The footman that will be attacking
     * @return The enemy footman ID this unit should attack
     */
    public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId) {
    	
    	int lastTurnNumber = stateView.getTurnNumber() - 1;
    	// There are still enemies to attack.
    	if (enemyFootmen.size() > 0) {
    		// It's the first turn.
    		if (lastTurnNumber < 0) {
    			// Return a randomly selected enemy to attack.
    			return enemyFootmen.get((int)random.nextDouble() * enemyFootmen.size());
    		}
    		// We are testing the policy.
    		else if (freezeForEvaluation && random.nextDouble() < epsilon) {
    			int selectedEnemyId = enemyFootmen.get(0);    			
    				
    			// Loop through all enemies and choose the one that maximizes the Q Value.
    			for (int i = 0; i < enemyFootmen.size(); i++) {
    				int tempEnemyId = enemyFootmen.get(i);
    				double tempQValue = calcQValue(stateView, historyView, attackerId, tempEnemyId);
    				
    				if (tempQValue > maxQValue) {
    					selectedEnemyId = tempEnemyId;
    				}
    			}
    			// Return the selected ID.
    			return selectedEnemyId;
    		}
    		// Choose the action that maximizes the Q value.
    		else {
    			int selectedEnemyId = enemyFootmen.get(0);
    			maxQValue = calcQValue(stateView, historyView, attackerId, selectedEnemyId);
    				
    			// Loop through all enemies and choose the one that maximizes the Q Value.
    			for (int i = 0; i < enemyFootmen.size(); i++) {
    				int tempEnemyId = enemyFootmen.get(i);
    				double tempQValue = calcQValue(stateView, historyView, attackerId, tempEnemyId);
    				
    				if (tempQValue > maxQValue) {
    					maxQValue = tempQValue;
    					selectedEnemyId = tempEnemyId;
    				}
    			}
    			// Return the selected ID.
    			return selectedEnemyId;
    		}
    	}
    	// We're out of enemies to attack.
    	return -1;
    }

    /**
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like:
     * 		Did this footman take or give damage? 
     * 		Did this footman die or kill its enemy. 
     * 		Did this footman start an action on the last turn? 
     * See the assignment description for the full list of rewards...
     *
     * Remember that you will need to discount this reward based on the timestep it is received on. 
     * See the assignment description for more details.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *     "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param stateView The current state of the game.
     * @param historyView History of the episode up until this turn.
     * @param footmanId The footman ID you are looking for the reward from.
     * @return The current reward
     */
    public double calculateReward(State.StateView stateView, History.HistoryView historyView, int footmanId) {
    	
    	double reward = 0.0;
    	int lastTurnNumber = stateView.getTurnNumber() - 1;
    	
    	// Check if it's the first turn and return.
    	if (lastTurnNumber < 0) return reward;    	
    	
    	/*	Check for dealt/received damage.
    	 *  If a friendly footman hits an enemy for d damage it gets +d reward
    	 *  If a friendly footman gets hit for d damage it gets -d penalty    
    	 */
    	for(DamageLog damageLog : historyView.getDamageLogs(lastTurnNumber)) {
//    		// TODO: Remove print statement.
//    	     System.out.println("Defending player: " + damageLog.getDefenderController() 
//    	    		 			+ " defending unit: " + damageLog.getDefenderID() 
//    	    		 			+ " attacking player: " + damageLog.getAttackerController() 
//    	    		 			+ "attacking unit: " + damageLog.getAttackerID());
    	     
    	     // This footman attacked an enemy.
    	     if (damageLog.getAttackerID() == footmanId) {
    	    	reward += damageLog.getDamage();
    	     }
    	     // This footman was attacked.
    	     else if (damageLog.getDefenderController() == footmanId) {
    	    	 reward -= damageLog.getDamage(); 
    	     }
    	}
    	
    	/* Check for units that were killed.
    	 * If an enemy footman is killed +100 reward
    	 * If a friendly footman gets killed -100 penalty
    	 */
    	for (DeathLog deathLog : historyView.getDeathLogs(lastTurnNumber)) {
    		int controllerID = deathLog.getController();
    		Integer deadUnitID = deathLog.getDeadUnitID();
    		
    		// An enemy unit was killed in the last turn.
    		if (controllerID == ENEMY_PLAYERNUM) {
    			Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, lastTurnNumber);
    			
    			/*
    			 * We need to check if this footman performed an attack action last 
    			 * turn and if it actually killed the enemy identified by deathLog.
    			 */
    			for (ActionResult ar : actionResults.values()) {
    				
    				// This footman performed the action and the target was the dead unit.
    				if (ar.getAction().getUnitId() == footmanId && ((TargetedAction)ar.getAction()).getTargetId() == deadUnitID) {
    					reward += 100;
    				}
    			}
    		}
    		// This footman was killed in the last turn.
    		else if (footmanId == deadUnitID) {
    			reward -= 100;
    		}
    	}
    	
    	// Each action costs the agent -0.1	
    	Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
    	
    	for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
        	// System.out.println("Unit " + commandEntry.getKey() + " was commanded to " + commandEntry.getValue().toString());
        	
        	// This footman received a command last turn.
        	// TODO: May need to check if the command is the same as the command issued from two previous turns...
        	if (commandEntry.getKey() == footmanId) {
        		reward -= 0.1;
        	}
        }
    	// Set the reward of the footman for testing evaluation.
    	if (freezeForEvaluation && footmanId < testingRewards.length) {
    		testingRewards[footmanId] = reward;
    	}
    	else {
    		testingRewards[myFootmen.size() - 1] = reward;
    	}
    	
        return reward;
    }

    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will calculate
     * your features and multiply them by your current weights to get the approximate Q-value.
     *
     * @param stateView Current SEPIA state
     * @param historyView Episode history up to this point in the game
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman that your footman would be attacking
     * @return The approximate Q-value
     */
    public double calcQValue(State.StateView stateView, History.HistoryView historyView, int attackerId, int defenderId) {
    	
    	double[] featureVector = calculateFeatureVector(stateView, historyView, attackerId, defenderId);
    	
    	// Exit if these vectors are not the same length.
    	if (weights.length != featureVector.length) System.exit(0);
    	
    	// Calculate the dot product of the weight vector and the feature vector.
    	double dotProduct = 0;
    	
    	for (int i = 0; i < featureVector.length; i++) {
    		dotProduct += featureVector[i] * weights[i];
    	}
    	// Return the Q value    	
    	return dotProduct + weights[0];	// TODO: Not sure if adding weights[0] is right...
    }

    /**
     * Given a state and action calculate your features here. Please include a comment explaining what features
     * you chose and why you chose them.
     *
     * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
     * take a dot product of this array with the weights array to get a Q-value for a given state action.
     *
     * It is a good idea to make the first value in your array a constant. This just helps remove any offset
     * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment description.
     *
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The array of feature function outputs.
     */
    public double[] calculateFeatureVector(State.StateView stateView, History.HistoryView historyView, int attackerId, int defenderId) {
        
    	double[] featureVector = new double[NUM_FEATURES];
    	int lastTurnNumber = stateView.getTurnNumber() - 1;
    	UnitView attacker = stateView.getUnit(attackerId);
    	UnitView defender = stateView.getUnit(defenderId);    	    	
    	
    	// Set the initial feature to a constant as suggested in assignment.
    	featureVector[0] = 0.5;
    	
    	if (attacker != null && defender != null) {
        	// Is the enemy the closest to attacker by Chebyshev distance?
    		featureVector[1] = (1 / getChebyshevDistance(attacker.getXPosition(), 
					attacker.getYPosition(), 
					defender.getXPosition(), 
					defender.getYPosition())) * 100;
    		
        	// Avoid enemies with higher health.
        	featureVector[2] = defender.getHP() > 0 ? attacker.getHP() / defender.getHP() : 1;
        	
        	// Is this enemy currently attacking me (the footman)?
        	Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, lastTurnNumber);
        	
        	// There were action results from the previous turn.
        	if (actionResults != null) {
        		
        		// One of the actions had to do with the defender.
        		if (actionResults.containsKey(defenderId)) {
        			TargetedAction targetedAction = (TargetedAction)actionResults.get(defenderId).getAction();
        			
        			if (targetedAction != null && targetedAction.getTargetId() == attackerId) {
            			featureVector[3] = 100;	// TODO: Not really sure what should be assigned here...
            		}
            		else {
                		featureVector[3] = 1;
                	}
        		}
        		else {
            		featureVector[3] = 1;
            	}
        		
        		// Is the enemy being attacked by at least one other footman already?
            	int numAttackers = 0;
            	
            	for (ActionResult ar : actionResults.values()) {
            		if (((TargetedAction)ar.getAction()).getTargetId() == defenderId) {
            			numAttackers++;
            		}
            	}
            	featureVector[4] = numAttackers > 0 ? (double)(1 / numAttackers) : 1;
        	}
    	}
    	// Either the attacker or defender was destroyed during this call.
    	else {
    		featureVector[1] = 0;
    		featureVector[2] = 0;
    		featureVector[3] = 0;
    		featureVector[4] = 0;
    	}
    	
    	return featureVector;
    }
    
    /**
     * Determines whether a significant event has occurred in the game.
     * @return
     */
    private boolean significantEvent(State.StateView stateView, History.HistoryView historyView) {    	
    	
    	int lastTurnNumber = stateView.getTurnNumber() - 1;
    	
    	// TODO: Remove
//    	System.out.println("Checking for events on turn: " + lastTurnNumber);
    	
    	// First turn
    	if (lastTurnNumber < 0) {
        	// TODO: Remove
//        	System.out.println("EVENT: First Turn");
    		
    		return true;
    	}
    	// Any footman killed
    	if (historyView.getDeathLogs(lastTurnNumber).size() > 0) {
        	// TODO: Remove
//        	System.out.println("EVENT: Footman killed");
    		
    		return true;
    	}
    	// Friendly footman attacked
    	for (DamageLog damageLog : historyView.getDamageLogs(lastTurnNumber)) {
    		if (myFootmen.contains(damageLog.getDefenderID())) {
            	// TODO: Remove
//            	System.out.println("EVENT: Allied footman attacked");
    			
    			return true;
    		}
    	}
    	Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, lastTurnNumber);
    	// Friendly footman action completed
    	for (ActionResult ar : actionResults.values()) {
    		if (myFootmen.contains(ar.getAction().getUnitId()) && ar.getFeedback().toString().equals("INCOMPLETE")) {
            	// TODO: Remove
//            	System.out.println("EVENT: Action completed");
    			
    			return true;
    		}
    	}
    	
    	return false;
    }
    
    /**
     * Helper method that calculates and updates the rewards for all footmen for a given StateView and HistoryView.
     * @param stateView
     * @param historyView
     */
    private void calculateFootmenRewards(State.StateView stateView, History.HistoryView historyView) {
    	    	
    	for (Integer id : myFootmen) {
    		double stateReward = calculateReward(stateView, historyView, id);
    		double currentTotalReward = footmenRewardMap.get(id); 
    		footmenRewardMap.put(id, currentTotalReward + stateReward);    		
    	}
    }
    
    /**
     * Helper method that removes any units killed in the last turn. 
     * @param stateView
     * @param historyView
     */
    private void removeDeadUnits(State.StateView stateView, History.HistoryView historyView) {
    	
    	for (DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() - 1)) {
			int controllerId = deathLog.getController();
    		Integer deadUnitID = deathLog.getDeadUnitID();
//			System.out.println("Player: " + controllerId + " | Unit " + deadUnitID + " was killed.");
			
			// Remove any of the player's units that were killed in the last turn.
			if (controllerId == playernum && myFootmen.contains(deadUnitID)) {
				cumulativeReward += footmenRewardMap.get(deadUnitID);	// Update the cumulative reward.
				myFootmen.remove(deadUnitID);
			}
			// Remove any of the enemy's units that were killed in the last turn.
			else if (controllerId == ENEMY_PLAYERNUM && enemyFootmen.contains(deadUnitID)) {
				enemyFootmen.remove(deadUnitID);
			}
			// An unidentified unit was killed and we don't know what to do with it.
			else {
				System.err.println("Unknown unit killed. Exiting with failure...");
				System.exit(0);
			}
		}
    }
    
    /**
     * Calculates the Chebyshev distance between two coordinates (x1, y1), (x2, y2).
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     * @return
     */
    private double getChebyshevDistance(int x1, int y1, int x2, int y2) {
    	
    	return Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1));
    }
    
    /*
     * ///////////////////////////////////////////////////////////////////////////////////////
     * DON'T MODIFY ANTHING BEYOND THIS POINT.
     * ///////////////////////////////////////////////////////////////////////////////////////
     */

    /**
     * DO NOT CHANGE THIS!
     *
     * Prints the learning rate data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    public void printTestData (List<Double> averageRewards) {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++) {
            String gamesPlayed = Integer.toString(10*i);
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++) {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will take your set of weights and save them to a file. Overwriting whatever file is
     * currently there. You will use this when training your agents. You will include the output of this function
     * from your trained agent with your submission.
     *
     * Look in the agent_weights folder for the output.
     *
     * @param weights Array of weights
     */
    public void saveWeights(Double[] weights) {
        File path = new File("agent_weights/weights.txt");
        // create the directories if they do not already exist
        path.getAbsoluteFile().getParentFile().mkdirs();

        try {
            // open a new file writer. Set append to false
            BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

            for (double weight : weights) {
                writer.write(String.format("%f\n", weight));
            }
            writer.flush();
            writer.close();
        } catch(IOException ex) {
            System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
        }
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
     * can be created using the saveWeights function. You will use this function if the load weights argument
     * of the agent is set to 1.
     *
     * @return The array of weights
     */
    public Double[] loadWeights() {
        File path = new File("agent_weights/weights.txt");
        if (!path.exists()) {
            System.err.println("Failed to load weights. File does not exist");
            return null;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line;
            List<Double> weights = new LinkedList<>();
            while((line = reader.readLine()) != null) {
                weights.add(Double.parseDouble(line));
            }
            reader.close();

            return weights.toArray(new Double[weights.size()]);
        } catch(IOException ex) {
            System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
        }
        return null;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }
}
