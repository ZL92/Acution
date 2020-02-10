#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:12:13 2019

@author: Chinmay, Sai, lizhaolin, Tom
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Initialize Seller profit and Buyer profit dictionaries
def _initSellersBuyers(n_sellers, n_buyers):
    seller_profits = {}
    buyer_profits = {}
    for k in range(n_sellers):
        seller_profits["S"+str(k+1)] = 0
    for n in range(n_buyers):
        buyer_profits["B"+str(n+1)] = 0

    # Seller profits = {'S1':0, 'S2':0, ...}
    # same for buyer profits

    return seller_profits, buyer_profits

# Update alpha for Buyer n and Seller k based on previous round
def _updateAlpha(alpha, n, k, R, HISTORY, bid_dec_factor, bid_inc_factor):
    E_k_prev = HISTORY['R'+str(R)][1][R-1, k] # Market proce of auction k in the previous round
    B_nk_prev = HISTORY['R'+str(R)][0][R-1, n, k] # Buyer n's bid in auction k in the previous round
    winning_bid_list = HISTORY['R'+str(R)][2]


    if alpha[n,k] == 1: ### Exp
        alpha[n,k] = 1
        return alpha


    if (B_nk_prev in winning_bid_list or B_nk_prev >= E_k_prev) and alpha[n,k] > 1:
        print("Decreasing alpha -- Buyer", n+1)
        temp = alpha[n,k]
        alpha[n,k] *=  bid_dec_factor

        if alpha[n,k] < 1:
            #alpha[n,k] = temp
            alpha[n,k] = temp - (temp - 1)/2
        #alpha[n,k] = max(alpha[n,k], 1)

    else:
        print("Increasing alpha -- Buyer", n+1)
        alpha[n,k] *=  bid_inc_factor

    return alpha

################################################################################
    ''' Auction functions '''
################################################################################
def pureAuction(n_sellers, n_buyers, n_rounds, S_max, bid_dec_factor=0.9, bid_inc_factor=1.1):
    seller_profits, buyer_profits = _initSellersBuyers(n_sellers, n_buyers)

    seller_IDs = list(seller_profits.keys()) # Seller IDs = ['S1', 'S2', ...]
    buyer_IDs = list(buyer_profits.keys())   # Buyer IDs = ['B1', 'B2', ...]

    alpha = np.random.random(size=(n_buyers, n_sellers)) * 10 + 1 # alpha is in range [1,10]
    #alpha[1,:] = 1 ### Exp
    B = np.zeros(shape=(n_rounds, n_buyers, n_sellers)) # Storing the bidding values of ALL buyers for ALL auctions over ALL rounds
                                                # Zero if the buyer didn't participate
    mrkt_prices = np.zeros(shape=(n_rounds, n_sellers))
    HISTORY = {}   # {'R1': (B, mrkt_prices, winning_bid_list), ...}  ----- Mainly used to update alpha every round

    for R in range(n_rounds): # Iterate over all ROUNDS
        sellers_list = seller_IDs.copy()
        np.random.shuffle(sellers_list)
        remaining_buyers = list(buyer_IDs.copy())
        print("\n--- Round", R+1, "---")
        #print("Bidding factor (alpha): \n", alpha)

        winning_bid_list = []  # WINNING BIDS of each auction in this round

        for s_id in sellers_list: # Iterate over all SELLERS -- Each seller organizes a single auction
            k = int(s_id[1:]) - 1
            S_k = np.random.randint(1, S_max+1) # Set the starting price of the auction
            print("\n- Auction organized by Seller", k+1, "-")
            print("> Participating Buyers: ", remaining_buyers)
            B_k = [] # Bids of buyers who PARTICIPATED in k

            for b_id in remaining_buyers: # Iterate over the REMAINING BUYERS
                n = int(b_id[1:])-1

                if R > 1:
                    alpha = _updateAlpha(alpha, n, k, R, HISTORY, bid_dec_factor, bid_inc_factor)
                B_nk = alpha[n, k] * S_k

                B_k.append(B_nk)
                B[R, n, k] = B_nk

            E_k = np.mean(B_k) # Calculate the market value for auction k
            mrkt_prices[R, k] = E_k  # Store it

            ##
            print("> All bids: ", B_k)
            print("> Market price: ", E_k)
            bids_below_mrkt_price = [b for b in B_k if b < E_k]
            print("> Bids below market price: ", bids_below_mrkt_price)
            bids_below_mrkt_price.sort(reverse=True) # Bids below market price
            winning_bid = bids_below_mrkt_price[0]   # Find the winning bid
            winner_buyer_idx = np.argwhere(B[R, :, k] == winning_bid)[0][0] # Determine the winner

            winning_bid_list.append(winning_bid)  # To be stored in HISTORY

            if len(bids_below_mrkt_price) > 1:
                winner_buyer_pays = bids_below_mrkt_price[1] # Winner pays 2nd bid (if it exists)
            elif len(bids_below_mrkt_price) == 1: # If there is NO 2nd bid
                winner_buyer_pays = (winning_bid + S_k) / 2 # Winner pays avg

            print("> Winning bid: ", winning_bid)
            print("> Winner: Buyer", winner_buyer_idx+1)
            print("> 2nd bid (Winner pays):", winner_buyer_pays)

            remaining_buyers.remove("B"+str(winner_buyer_idx+1)) # This winner will not participate in the remaining auctions of this round

            # Calculate profits
            seller_profits[s_id] += winner_buyer_pays
            buyer_profits["B"+str(winner_buyer_idx+1)] += E_k - winner_buyer_pays

        HISTORY['R'+str(R+1)] = tuple([B, mrkt_prices, winning_bid_list])

        #if R == n_rounds-1:
        #    print("Round: ", R+1)
        #    print("Bidding factor (alpha): \n", alpha)
            ##
    print("\n----- -----")
    #print("\nMarket prices: ", mrkt_prices)
    #print("\nB :", B)
    return mrkt_prices, seller_profits, buyer_profits




def LCAuction(n_sellers, n_buyers, n_rounds, S_max, epsilon, bid_dec_factor=0.9, bid_inc_factor=1.1):
    seller_profits, buyer_profits = _initSellersBuyers(n_sellers, n_buyers)

    seller_IDs = list(seller_profits.keys())
    buyer_IDs = list(buyer_profits.keys())

    alpha = np.random.random(size=(n_buyers, n_sellers)) * 10 + 1 # alpha is in range [1,10]
    B = np.zeros(shape=(n_rounds, n_buyers, n_sellers)) # Storing the bidding values of ALL buyers for ALL auctions over ALL rounds
                                                # Zero if the buyer didn't participate
    mrkt_prices = np.zeros(shape=(n_rounds, n_sellers))

    HISTORY = {}   # {'R1': (B, mrkt_prices, winning_bid_list), ...}  ----- Mainly used to update alpha

    ###   AUCTION   ### --------------------------------------------------------
    for R in range(n_rounds):
        sellers_list = seller_IDs.copy()
        np.random.shuffle(sellers_list)
        print("\n--- Round", R+1, "---")
        #print("Bidding factor (alpha): \n", alpha)

        auction_list = [] # Ordered of sellers who orgarized auction in this round
        winning_bid_list = [] # WINNING BIDS of each auction in this round
        second_highest_bid_list = [] # Ordered list of the 2nd highest bid values (which the winner has to pay)
        winner_buyer_idx_list = [] # Ordered list of indices of the winning BUYERS in this round
        winner_profit_list = [] # Ordered list of profits of winners of all the auctions of this round

        winner_data = {}  # {Auction #: (winning_bid, winner_buyer_idx, winner_profit)}

        for s_id in sellers_list: # Each seller organizes a single auction
            k = int(s_id[1:]) - 1
            S_k = np.random.randint(1, S_max+1) # Set the starting price of the auction
            print("\n-- Auction organized by Seller", k+1, "--")
            print("> Participating Buyers: ", buyer_IDs)
            B_k = [] # Bids of buyers who PARTICIPATED in k

            for n in range(n_buyers):

                # If this buyer had won an earlier auction in this round
                if n in winner_buyer_idx_list:

                    list_idxs = [i for i, x in enumerate(winner_buyer_idx_list) if x == n]
                    list_idx = max(list_idxs)

                    prev_paid_bid_value = second_highest_bid_list[list_idx]
                    corresponding_seller = auction_list[list_idx]

                    penalty = epsilon * prev_paid_bid_value

                    E_x = mrkt_prices[R, int(corresponding_seller[1])-1]

                    # Calculate this buyer's bid
                    if R > 1:
                        alpha = _updateAlpha(alpha, n, k, R, HISTORY, bid_dec_factor, bid_inc_factor)

                    B_nk = alpha[n, k] * S_k - (E_x - prev_paid_bid_value + penalty)
                    if B_nk < 0: # If negative, then this buyer will NOT bid
                        B_nk = 'nil'
                        print("Buyer", n+1, "doesn't bid since it's current bid is negative -- Bidding not profitable")

                    #print("### E_x:",E_x)
                    #print("### prev paid amount:",prev_paid_bid_value)
                    #print("### B_nk:",B_nk)

                else:
                    if R > 1:
                        alpha = _updateAlpha(alpha, n, k, R, HISTORY, bid_dec_factor, bid_inc_factor)
                    B_nk = alpha[n, k] * S_k

                B_k.append(B_nk)
                if 'nil' in B_k:
                    B_k.remove('nil')
                    B_nk = 0
                B[R, n, k] = B_nk

            E_k = np.mean(B_k) # Calculate the market value for auction k
            mrkt_prices[R, k] = E_k  # Store it

            ##
            print("> All bids: ", B_k)
            print("> Market price: ", E_k)
            bids_below_mrkt_price = [b for b in B_k if b < E_k]
            print("> Bids below market price: ", bids_below_mrkt_price)
            bids_below_mrkt_price.sort(reverse=True) # Bids below market price
            winning_bid = bids_below_mrkt_price[0]   # Find the winning bid
            winner_buyer_idx = np.argwhere(B[R, :, k] == winning_bid)[0][0] # Determine the winner

            if len(bids_below_mrkt_price) > 1:
                winner_buyer_pays = bids_below_mrkt_price[1] # Winner pays 2nd bid (if it exists)
            elif len(bids_below_mrkt_price) == 1: # If there is NO 2nd bid
                winner_buyer_pays = (winning_bid + S_k) / 2 # Winner pays avg

            print("> Winning bid: ", winning_bid)
            print("> Winner: Buyer", winner_buyer_idx+1)
            print("> 2nd bid (Winner pays):", winner_buyer_pays)

            # If the winner buyer had won previously in the same round
            if winner_buyer_idx in winner_buyer_idx_list:
                list_idxs = [i for i, x in enumerate(winner_buyer_idx_list) if x == winner_buyer_idx]
                list_idx = max(list_idxs)

                winner_prev_profit = np.array(winner_profit_list)[list_idx]
                winner_curr_profit = E_k - winner_buyer_pays

                # If winner's previous profit is less than his current profit
                if winner_prev_profit < winner_curr_profit:
                    # Decommit to previous seller

                    least_profitable_bid_value = second_highest_bid_list[list_idx]
                    corresponding_seller = auction_list[list_idx]

                    penalty = epsilon * least_profitable_bid_value

                    # Calculate profits --
                    # 1. Profit of the seller to whom the winner decomitted
                    seller_profits[corresponding_seller] = seller_profits[corresponding_seller] - least_profitable_bid_value + penalty
                    # 2. Profit of the current seller
                    seller_profits[s_id] += winner_buyer_pays
                    # 3. Profit of the winner buyer
                    buyer_profits["B"+str(winner_buyer_idx+1)] = buyer_profits["B"+str(winner_buyer_idx+1)] \
                                                                 + (E_k - winner_buyer_pays) \
                                                                 - penalty
                # Else if his current profit is less than from his previous auction
                elif winner_prev_profit > winner_curr_profit:
                    # Decommit to the current seller

                    penalty = epsilon * winner_buyer_pays

                    prev_paid_bid_value = second_highest_bid_list[list_idx]

                    # Calculate profits --
                    # 1. Profit of the current seller
                    seller_profits[s_id] = seller_profits[s_id] - winner_buyer_pays + penalty
                    # 2. Profit of the winner buyer
                    buyer_profits["B"+str(winner_buyer_idx+1)] = buyer_profits["B"+str(winner_buyer_idx+1)] \
                                                                 - penalty

                print("> Buyer", winner_buyer_idx+1,"decommits with Seller",corresponding_seller[1])
                print("> Penalty:", penalty)

            else: # If the buyer has won for the first time in this round
                # Calculate profits
                seller_profit = winner_buyer_pays
                seller_profits[s_id] += seller_profit

                winner_profit = E_k - winner_buyer_pays
                buyer_profits["B"+str(winner_buyer_idx+1)] += winner_profit


            # Update the auction data of this round
            auction_list.append(s_id)
            winning_bid_list.append(winning_bid)
            second_highest_bid_list.append(winner_buyer_pays)
            winner_buyer_idx_list.append(winner_buyer_idx)
            winner_profit_list.append(winner_profit)

            winner_data[s_id] = (winner_buyer_pays, winner_buyer_idx, winner_profit)

        HISTORY['R'+str(R+1)] = tuple([B, mrkt_prices, winning_bid_list])
            ##
    print("\n----- -----")
    #print("\nMarket prices: ", mrkt_prices)
    #print("\nB :", B)

    return mrkt_prices, seller_profits, buyer_profits


# ------------------------------------------------------------------------------

# Auction selector function
def simulateAuction(auction_type, n_sellers, n_buyers, n_rounds, S_max, epsilon):

    if auction_type == 'p':
        return pureAuction(n_sellers, n_buyers, n_rounds, S_max)

    elif auction_type == 'lc':
        return LCAuction(n_sellers, n_buyers, n_rounds, S_max, epsilon)

################################################################################
    ''' Main function '''
################################################################################
def main(n_sellers, n_buyers, n_rounds, S_max, epsilon, auction_type):
    print("-------------- Auction Simulation --------------\n\nEnter the parameters --")

    # For final implementation -- Take inputs from users
    '''
    n_sellers = int(input(">> Number of sellers: "))
    n_buyers = int(input(">> Number of buyers: "))     # n_buyers > n_sellers
    n_rounds = int(input(">> Number of rounds: "))
    S_max = int(input(">> Max starting price: "))
    epsilon = float(input(">> Penalty factor: "))
    auction_type = input("Auction type - pure/leveled commitment (p/lc): ")
    '''

    print("Sellers:", n_sellers, "| Buyers:", n_buyers, "| Rounds:", n_rounds)

    # Simulate the auction
    mrkt_prices, seller_profits, buyer_profits = simulateAuction(auction_type,
                                                                  n_sellers, n_buyers,
                                                                  n_rounds,
                                                                  S_max,
                                                                  epsilon)

    # Returned data --
    # 1. Market Prices (numpy array):  [[E_1, E_2, E_3],  -- round 1, all auctions
    #                                   [E_1, E_2, E_3],  -- round 2, all auctions
    #                                   [E_1, E_2, E_3]]  -- round 3, all auctions
    #
    # 2. Seller profits (dict): {S1:ps_1, S2:ps_2, S3:ps_3}  -- Total profit of each seller
    #
    # 3. Buyer profits (dict): {B1:pb_1, B2:pb_2, B3:pb_3, B4:pb_4, B5:pb_5} -- Total profit of each buyer

    #print("Market Prices: ", mrkt_prices) # Need to plot this
    '''
    legend = []
    n_sellers_list = range(1, n_sellers+1)
    for r in range(n_rounds):
        plt.plot(n_sellers_list, mrkt_prices[r])
        legend.append("Round "+str(r+1))
    plt.title("Market Prices")
    plt.xlabel("Auction #")
    plt.ylabel("Market price")
    plt.legend(legend)
    plt.show()
    '''
    legend = []
    n_rounds_list = range(1, n_rounds+1)
    for k in range(n_sellers):
        plt.plot(n_rounds_list, mrkt_prices[:,k])
        legend.append("Seller "+str(k+1))
    plt.plot(n_rounds_list, S_max*np.ones(shape=(len(n_rounds_list),1)), 'k--')
    legend.append("S_max")

    plt.title("Market Prices across rounds")
    plt.xlabel("Round")
    plt.ylabel("Market price")
    plt.legend(legend)
    plt.show()

    print("Sellers' profits: ", seller_profits)
    print("Buyers' profits: ", buyer_profits)
    total_seller_profits=0
    for value in seller_profits.values():
        total_seller_profits += value
    print("Average Sellers' profits/round: ", total_seller_profits/(n_sellers*n_rounds))
    total_buyer_profits=0
    for value in buyer_profits.values():
        total_buyer_profits += value    
    print("Average Buyers' profits/round: ", total_buyer_profits/(n_buyers*n_rounds))
#    print("Market price: ", mrkt_prices)
#    print('Average market price: ', np.mean(mrkt_prices))
    height=list(seller_profits.values())
    bars=list(seller_profits.keys())
    y_pos=np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Seller Profit")
    plt.xlabel("Sellers")
    plt.ylabel("Profit")
    plt.show()
    
    height=list(buyer_profits.values())
    bars=list(buyer_profits.keys())
    y_pos=np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Buyer Profit")
    plt.xlabel("Buyers")
    plt.ylabel("Profit")
    plt.show()

################################################################################
    ''' Run '''
################################################################################
# Set parameters for experiments
n_sellers = 3
n_buyers = 5
n_rounds = 200
S_max = 10
epsilon = 0.2
auction_type = 'p'   # 'p' for Pure ; 'lc' for Leveled Commitment

main(n_sellers, n_buyers, n_rounds, S_max, epsilon, auction_type)
