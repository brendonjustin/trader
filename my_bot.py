import random

import traders
import run_experiments
import plot_simulation
import other_bots

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
        self.useAvg = 0
    
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

    def find_best_quantity(self, quantity, stride, tradeString, check_callback):
        if quantity == 0:
            return 0, 0

        stride = stride / 2

        if stride == 0:
            if tradeString == 'buy':
                return quantity, quantity*(self.useAvg - check_callback(tradeString, quantity))
            else:
                return quantity, quantity*(check_callback(tradeString, quantity) - self.useAvg)

        qtyL = quantity-stride
        qtyR = quantity+stride
        profitL = 0
        profitR = 0

        if qtyL > 0:
            qtyL, profitL = self.find_best_quantity(quantity-stride, stride, tradeString, check_callback)
        if qtyR > 0:
            qtyR, profitR = self.find_best_quantity(quantity+stride, stride, tradeString, check_callback)

        if profitL > profitR:
            return qtyL, profitL
        else:
            return qtyR, profitR

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
        jumpThreshold = 0.15

        #   Don't trade on very limited information
        if len(self.information) < 10:
            return

        # Consider the largest difference between two moving averages
        # to be a jump point if said difference is greater than some theshold
        if len(self.information) > 2*windowSize:
            # movingAvg = numpy.average(self.information[-windowSize:])
            # preMovingAvg = numpy.average(self.information[-2*windowSize:-windowSize])

            # harmonicDenom = numpy.sum(map(lambda x : float(1) / (windowSize - x), range(windowSize)))
            harmonicDenom = 1

            # use harmonically weighted averages of the two moving windows
            movingAvg = numpy.sum(map(lambda x, y : float(y) / (windowSize - x), 
                range(windowSize), self.information[-windowSize:]))
            movingAvg = movingAvg / harmonicDenom

            preMovingAvg = numpy.sum(map(lambda x, y : float(y) / (windowSize - x), 
                range(windowSize), reversed(self.information[-2*windowSize:-windowSize])))
            preMovingAvg = preMovingAvg / harmonicDenom

            # print "harmonicDenom:", harmonicDenom, " movingAvg:", movingAvg, " preMovingAvg:", preMovingAvg

            self.diffMovingAvg.append(movingAvg-preMovingAvg)

            if self.diffMovingAvg[-1] == max(self.diffMovingAvg) and abs(self.diffMovingAvg[-1]) > jumpThreshold:
                self.lastJumpIndex = len(self.diffMovingAvg) + windowSize
                # print "Jumping at:", self.lastJumpIndex, " diffMovingAvg:", self.diffMovingAvg[-1]

        avg = numpy.average(self.information[self.lastJumpIndex:])

        self.useAvg = avg*100

        buyDiff = 0
        sellDiff = 0
        bestBuyQuantity = 0
        bestSellQuantity = 0
        bestQuantity = 0
        rangeStride = max(2*len(self.information[self.lastJumpIndex:]), maxUncertainQuantity) / 2
        quantity = rangeStride
        bestBuyQuantity, buyDiff = self.find_best_quantity(quantity, rangeStride, 'buy', check_callback)
        bestSellQuantity, sellDiff = self.find_best_quantity(quantity, rangeStride, 'sell', check_callback)

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

def main():
    bots = [MyBot()]
    # bots.extend(other_bots.get_bots(5, 2))
    # bots.extend(other_bots.get_bots(num_fundamentals, num_technical))
    
    # Plot a single run. Useful for debugging and visualizing your
    # bot's performance. Also prints the bot's final profit, but this
    # will be very noisy.
    # plot_simulation.run(bots)
    
    # Calculate statistics over many runs. Provides the mean and
    # standard deviation of your bot's profit.
    run_experiments.run(bots, num_processes=3, simulations=2000, lmsr_b=150)

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
