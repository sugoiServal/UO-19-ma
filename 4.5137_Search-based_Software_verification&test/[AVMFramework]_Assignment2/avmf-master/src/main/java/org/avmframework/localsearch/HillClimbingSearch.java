package org.avmframework.localsearch;

import org.avmframework.TerminationException;
import org.avmframework.objective.ObjectiveValue;
import java.util.Random; 
public class HillClimbingSearch extends LocalSearch {

  public static int NEIGHBORHOOD_SIZE_DEFAULT = 50;
  public static int NEIGHBORHOOD_WINDOW_DEFAULT = 100;

  protected int neighborhoodSize = NEIGHBORHOOD_SIZE_DEFAULT;
  protected int neighborhoodWindow = NEIGHBORHOOD_WINDOW_DEFAULT;

  protected ObjectiveValue initial;
  protected ObjectiveValue newfit;
  protected ObjectiveValue last;
  protected ObjectiveValue next;
  protected int modifier;
  protected int num;
  protected int dir;

  public HillClimbingSearch() {}

  public HillClimbingSearch(int neighborhoodSize, int neighborhoodWindow) {
    this.neighborhoodSize = neighborhoodSize;
    this.neighborhoodWindow = neighborhoodWindow;
  }

  protected void performSearch() throws TerminationException {
	System.out.println("performing HillClimbing");  
    initialize();
    hillClimbing();
  }

  protected void initialize() throws TerminationException {
    initial = objFun.evaluate(vector);
    modifier = 1;
    num = var.getValue();
    dir = 0;
  }

  protected boolean establishDirection() throws TerminationException {
    // evaluate left move
    var.setValue(num - modifier);
    ObjectiveValue left = objFun.evaluate(vector);

    // evaluate right move
    var.setValue(num + modifier);
    ObjectiveValue right = objFun.evaluate(vector);

    // find the best direction
    boolean leftBetter = left.betterThan(initial);
    boolean rightBetter = right.betterThan(initial);
    if (leftBetter) {
      dir = -1;
    } else if (rightBetter) {
      dir = 1;
    } else {
      dir = 0;
    }

    // set num and the variable according to the best edge
    num += dir * modifier;
    var.setValue(num);

    // set last and next objective values
    last = initial;
    if (dir == -1) {
      next = left;
    } else if (dir == 1) {
      next = right;
    } else if (dir == 0) {
      next = initial;
    }

    return dir != 0;
}

  protected void hillClimbing() throws TerminationException {
	Random rand = new Random(); 
	last = initial;  //the first vector's eval
	next = initial;
	//for different batchs of neighborhoods
     do{  		//if there is an improvement over the previous batch
      last = next;
      int best_modifier = 0;      
      //generate neighborhoods
      for(int i =0; i<neighborhoodSize; i++) {
    	  int rand_int = rand.nextInt(neighborhoodWindow)-(neighborhoodWindow/2); 
    	  var.setValue(num + rand_int);
    	  newfit = objFun.evaluate(vector);				//generate one sol and evaluate 
    	  if (newfit.betterThan(next)){
    		  best_modifier = rand_int;
    		  next = newfit;   		  
    	  }
      }
      num += best_modifier;
      var.setValue(num);
    }while (next.betterThan(last));
  }
}
