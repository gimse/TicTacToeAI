# Tic Tac Toe
import random
import Ai
import numpy as np

def drawBoard(board):
    # This function prints out the board that it was passed.
    # "board" is a list of 10 strings representing the board (ignore index 0)
    print('   |   |')
    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('   |   |')
def inputPlayerLetter():
    # Lets the player type which letter they want to be.
    # Returns a list with the player’s letter as the first item, and the computer's letter as the second.
    letter = ''
    while not (letter == 'X' or letter == 'O'):
        print('Do you want to be X or O?')
        letter = input().upper()
    # the first element in the list is the player’s letter, the second is the computer's letter.
    if letter == 'X':
        return ['X', 'O']
    else:
        return ['O', 'X']
def whoGoesFirst():
    # Randomly choose the player who goes first.
    if random.randint(0, 1) == 0:
        return 'computer'
    else:
        return 'player'
def playAgain():
    # This function returns True if the player wants to play again, otherwise it returns False.
    print('Do you want to play again? (yes or no)')
    return input().lower().startswith('y')
def makeMove(board, letter, move):
    board[move] = letter
def isWinner(bo, le):
    # Given a board and a player’s letter, this function returns True if that player has won.
    # We use bo instead of board and le instead of letter so we don’t have to type as much.
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or # across the top
    (bo[4] == le and bo[5] == le and bo[6] == le) or # across the middle
    (bo[1] == le and bo[2] == le and bo[3] == le) or # across the bottom
    (bo[7] == le and bo[4] == le and bo[1] == le) or # down the left side
    (bo[8] == le and bo[5] == le and bo[2] == le) or # down the middle
    (bo[9] == le and bo[6] == le and bo[3] == le) or # down the right side
    (bo[7] == le and bo[5] == le and bo[3] == le) or # diagonal
    (bo[9] == le and bo[5] == le and bo[1] == le)) # diagonal
def getBoardCopy(board):
    # Make a duplicate of the board list and return it the duplicate.
    dupeBoard = []
    for i in board:
        dupeBoard.append(i)
    return dupeBoard
def isSpaceFree(board, move):
    # Return true if the passed move is free on the passed board.
    return board[move] == ' '
def getPlayerMove(board):
    # Let the player type in their move.
    move = ' '
    while move not in '1 2 3 4 5 6 7 8 9'.split() or not isSpaceFree(board, int(move)):
        print('What is your next move? (1-9)')
        move = input()
    return int(move)

def isBoardFull(board):
    # Return True if every space on the board has been taken. Otherwise return False.
    for i in range(1, 10):
        if isSpaceFree(board, i):
            return False
    return True


print('Welcome to Tic Tac Toe!')

ai1 =Ai.Ai(0.1,'X','O')
ai2 =Ai.Ai(0,1,'O','X')

learningrate=1
ai1.set_learningrate(learningrate)
ai2.set_learningrate(learningrate)

iterations=0

while True:
    # Reset the board
    theBoard = [' '] * 10
    playerLetter='X'
    computerLetter='O'
    turn = whoGoesFirst()
    #print('The ' + turn + ' will go first.')
    gameIsPlaying = True
    while gameIsPlaying:
        if turn == 'player':
            # Player’s turn.
            #drawBoard(theBoard)
            move = ai2.move(theBoard)

            makeMove(theBoard, playerLetter, move)

            ai1.learn(theBoard)

            if isWinner(theBoard, playerLetter):
                #drawBoard(theBoard)
                #print('Hooray! You have won the game!')
                ai1.reward(-99)
                ai2.reward(101)
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    #drawBoard(theBoard)
                    #print('The game is a tie!')
                    break
                else:
                    turn = 'computer'
        else:
            # Computer’s turn.
            move=ai1.move(theBoard)#!!!!!!!!!!!!!!!!!!!!Moves

            makeMove(theBoard, computerLetter, move)

            ai2.learn(theBoard)
            if isWinner(theBoard, computerLetter):
                #drawBoard(theBoard)
                #print('The computer has beaten you! You lose.')
                ai1.reward(99)
                ai2.reward(-101)
                gameIsPlaying = False
            else:
                if isBoardFull(theBoard):
                    #drawBoard(theBoard)
                    #print('The game is a tie!')
                    break
                else:
                    turn = 'player'
    if iterations>=5000000:
        ai1.save()
        break
    if np.mod(iterations,100)==0:
        print('iteration',iterations)
    if np.mod(iterations, 10000) == 0:
        ai1.save()
        learningrate=learningrate/1.5
        print('Learning rate: ',learningrate)
        ai1.set_learningrate(learningrate)
        ai2.set_learningrate(learningrate)
    iterations+=1