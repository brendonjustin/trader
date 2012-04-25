import random

import traders
import run_experiments
import plot_simulation

import numpy

class MyBot(traders.Trader):
    name = 'my_bot'

    def simulation_params(self, timesteps,
                          possible_jump_locations,
                          single_jump_probability):
        """Receive information about the simulation."""
        # Number of trading opportunities
        self.timesteps = timesteps
        # A list of timesteps when there could be a jump
        self.possible_jump_locations = possible_jump_locations
        # For each of the possible jump locations, the probability of
        # actually jumping at that point. Jumps are normally
        # distributed with mean 0 and standard deviation 0.2.
        self.single_jump_probability = single_jump_probability
        # A place to store the information we get
        self.information = []

        self.lastJumpIndex = 0
        self.diffMovingAvg = []
    
    def new_information(self, info, time):
        """Get information about the underlying market value.
        
        info: 1 with probability equal to the current
          underlying market value, and 0 otherwise.
        time: The current timestep for the experiment. It
          matches up with possible_jump_locations. It will
          be between 0 and self.timesteps - 1."""
        self.information.append(info)

    def trades_history(self, trades, time):
        """A list of everyone's trades, in the following format:
        [(execution_price, 'buy' or 'sell', quantity,
          previous_market_belief), ...]
        Note that this isn't just new trades; it's all of them."""
        self.trades = trades

    def trading_opportunity(self, cash_callback, shares_callback,
                            check_callback, execute_callback,
                            market_belief):
        """Called when the bot has an opportunity to trade.
        
        cash_callback(): How much cash the bot has right now.
        shares_callback(): How many shares the bot owns.
        check_callback(buysell, quantity): Returns the per-share
          price of buying or selling the given quantity.
        execute_callback(buysell, quantity): Buy or sell the given
          quantity of shares.
        market_belief: The market maker's current belief.

        Note that a bot can always buy and sell: the bot will borrow
        shares or cash automatically.
        """
        #   Some default values
        bestAction = 'buy'
        maxUncertainQuantity = 20
        windowSize = 20

        #   Don't trade on very limited information
        if len(self.information) < 10:
            return

        # print "zero"
        # Consider the largest difference between two moving averages
        # to be a jump point if said difference is greater than some theshold
        if len(self.information) > 2*windowSize:
            # print "a"
            movingAvg = numpy.average(self.information[-windowSize:])
            # print "b"
            preMovingAvg = numpy.average(self.information[-2*windowSize:-windowSize])
            # print "c"
            self.diffMovingAvg.append(movingAvg-preMovingAvg)

            if self.diffMovingAvg[-1] == max(self.diffMovingAvg) and abs(self.diffMovingAvg[-1]) > 0.3:
                self.lastJumpIndex = len(self.diffMovingAvg) + windowSize
                # print "Jumping at:", self.lastJumpIndex
            # print "e"

        # print "last jump index:", self.lastJumpIndex
        avg = numpy.average(self.information[self.lastJumpIndex:])
        # print "one"

        useAvg = avg*100

        buyDiff = 0
        sellDiff = 0
        bestBuyQuantity = 0
        bestSellQuantity = 0
        bestQuantity = 0
        for quantity in range(1, max(2*len(self.information[self.lastJumpIndex:]), maxUncertainQuantity)):
            cost = quantity*check_callback('buy', quantity)
            if useAvg*quantity - cost > buyDiff:
                buyDiff = useAvg*quantity - cost
                bestBuyQuantity = quantity

            gains = quantity*check_callback('sell', quantity)
            if gains - useAvg*quantity > sellDiff:
                sellDiff = gains - useAvg*quantity
                bestSellQuantity = quantity


            if sellDiff > buyDiff:
                bestQuantity = bestSellQuantity
                bestAction = 'sell'
            else:
                bestQuantity = bestBuyQuantity
                bestAction = 'buy'

        # Trade in the best way, if a beneficial trade was found
        if bestQuantity > 0:
            # print "Buying or selling? " + bestAction
            execute_callback(bestAction, bestQuantity)

        # Place a randomly sized trade in the direction of
        # our cumulative information.
        # quantity = random.choice(xrange(1, 100))
        # if (check_callback('buy', quantity) < 100*avg):
        #     execute_callback('buy', quantity)
        # elif check_callback('sell', quantity) > 100*avg:
        #     execute_callback('sell', quantity)

def main():
    bots = [MyBot()]
    
    # Plot a single run. Useful for debugging and visualizing your
    # bot's performance. Also prints the bot's final profit, but this
    # will be very noisy.
    plot_simulation.run(bots)
    
    # Calculate statistics over many runs. Provides the mean and
    # standard deviation of your bot's profit.
    # run_experiments.run(bots, num_processes=3, simulations=13)

# Extra parameters to plot_simulation.run:
#   timesteps=100, lmsr_b=150

# Extra parameters to run_experiments.run:
#   timesteps=100, num_processes=2, simulations=2000, lmsr_b=150

# Descriptions of extra parameters:
# timesteps: The number of trading rounds in each simulation.
# lmsr_b: LMSR's B parameter. Higher means prices change less,
#           and the market maker can lose more money.
# num_processes: In general, set this to the number of cores on your
#                  machine to get maximum performance.
# simulations: The number of simulations to run.

if __name__ == '__main__': # If this file is run directly
    main()
