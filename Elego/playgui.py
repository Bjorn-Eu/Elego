'''
This file implements a simple gui made in PyGame
for playing capture go against the computer.
'''

import pygame
import sys
import torch
from pygame.locals import *
from queue import Queue
from agent import Agent
from zagent import ZAgent
import gameplay.config_board as config_board
from gameplay.gamestate import GameState
from gameplay.board import Board
from gameplay.move import Move
from encoders.extendedencoder import ExtendedEncoder


class BoardDisplay:
    def __init__(self,size = 9):
        self.size = size
        self.is_active = True

        #declare constants
        self.screen_size = 480
        self.inner_board_size = 380
        self.INNER_BORDER = 80
        self.grid_size = 40
        self.WHITE = (255,255,255)
        self.GREY = (30,50,30)
        self.BLACK = (0,0,0)
        self.WOOD = (161, 102, 47)
        

        #create graphics objects
        pygame.init()
        pygame.display.set_caption('Elego | CaptureGo')
        self.DISPLAYSURF =  pygame.display.set_mode((self.screen_size, self.screen_size))

        self.DISPLAYSURF.fill(self.GREY)
        self.BIGFONT = pygame.font.Font('freesansbold.ttf', 25)
        self.SMALLFONT = pygame.font.Font('freesansbold.ttf',20)


        self.newGameSurf = self.BIGFONT.render('New Game',True, self.WHITE, self.GREY)
        self.newGameRect = self.newGameSurf.get_rect()
        self.newGameRect.topright = (self.screen_size/2+self.newGameRect.width/2, self.screen_size-40)

        self.yourturnSurf = self.SMALLFONT.render('Your turn',True,self.BLACK,self.GREY)
        self.yourturnRect = self.yourturnSurf.get_rect()
        self.yourturnRect.topright = (self.screen_size/2+self.yourturnRect.width/2,20)

        self.winSurf = self.SMALLFONT.render('You win!',True,self.BLACK,self.GREY)
        self.winRect = self.winSurf.get_rect()
        self.winRect.topright = (self.screen_size/2+self.winRect.width/2,20)

        self.loseSurf = self.SMALLFONT.render('You lose...',True,self.BLACK,self.GREY)
        self.loseRect = self.loseSurf.get_rect()
        self.loseRect.topright = (self.screen_size/2+self.loseRect.width/2,20)

        #initiate capture go instance
        self.board = config_board.atari_go9x9()
        self.gamestate = GameState(1,self.board)
        self.znet = torch.jit.load('nets9x9//znet38.pt')
        self.encoder = ExtendedEncoder(size=self.size)
        self.zagent = ZAgent(self.znet,self.size,self.encoder,root_noise=True,playouts=1000)



    def clear_text(self):
        pygame.draw.rect(self.DISPLAYSURF,self.GREY,(0,0,self.screen_size,50))
        

    def init_new_game(self):
        self.board = config_board.atari_go9x9()
        self.gamestate = GameState(1,self.board)
        self.is_active = True
        self.drawboard()
        
    #draws the background board and starting position
    def drawboard(self):
        self.DISPLAYSURF.fill(self.GREY)
        self.DISPLAYSURF.blit(self.yourturnSurf,self.yourturnRect)
        self.DISPLAYSURF.blit(self.newGameSurf, self.newGameRect)            
        pygame.draw.rect(self.DISPLAYSURF,self.WOOD,(self.INNER_BORDER-30,self.INNER_BORDER-30,self.inner_board_size,self.inner_board_size))
        for i in range(9):
            pygame.draw.line(self.DISPLAYSURF,self.BLACK,(self.INNER_BORDER+self.grid_size*i,self.INNER_BORDER),(self.INNER_BORDER+self.grid_size*i,400))
            pygame.draw.line(self.DISPLAYSURF,self.BLACK,(self.INNER_BORDER,self.INNER_BORDER+40*i),(400,self.INNER_BORDER+self.grid_size*i))   

        #draw markers
        marker_size = 4
        pygame.draw.circle(self.DISPLAYSURF,self.BLACK,(self.INNER_BORDER+self.grid_size*4,self.INNER_BORDER+4*self.grid_size),marker_size)
        pygame.draw.circle(self.DISPLAYSURF,self.BLACK,(self.INNER_BORDER+self.grid_size*2,self.INNER_BORDER+2*self.grid_size),marker_size)
        pygame.draw.circle(self.DISPLAYSURF,self.BLACK,(self.INNER_BORDER+self.grid_size*6,self.INNER_BORDER+6*self.grid_size),marker_size)
        pygame.draw.circle(self.DISPLAYSURF,self.BLACK,(self.INNER_BORDER+self.grid_size*2,self.INNER_BORDER+6*self.grid_size),marker_size)
        pygame.draw.circle(self.DISPLAYSURF,self.BLACK,(self.INNER_BORDER+self.grid_size*6,self.INNER_BORDER+2*self.grid_size),marker_size)

        #draw starting position
        self.drawstone(self.BLACK,4,4)
        self.drawstone(self.BLACK,3,3)
        self.drawstone(self.WHITE,3,4)
        self.drawstone(self.WHITE,4,3)


    def drawstone(self,color,x,y):
        pygame.draw.circle(self.DISPLAYSURF, color, (self.INNER_BORDER+self.grid_size*x, self.INNER_BORDER+self.grid_size*y),16,0)

    #runs the main game loop
    def run(self):
        while True:
            if (not self.gamestate.is_over()) and self.gamestate.turn == -1:
                move = self.zagent.select_move(self.gamestate)
                if self.gamestate.board.is_capture(move):
                    self.clear_text()
                    self.DISPLAYSURF.blit(self.loseSurf,self.loseRect)
                    self.is_active = False
                self.gamestate.move(move)
                self.drawstone(self.WHITE,move.x-1,move.y-1)
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONUP:
                    x_mouse, y_mouse = event.pos
                    if self.newGameRect.collidepoint(x_mouse,y_mouse):
                        self.init_new_game()
                    x, y = self.find_index(x_mouse,y_mouse)

                    if x >=0 and x <self.size and y>=0 and y<self.size:
                        move = Move(1,x+1,y+1)
                        if self.is_active and self.gamestate.is_legal_move(move):
                            if self.gamestate.board.is_capture(move):
                                self.clear_text()
                                self.DISPLAYSURF.blit(self.winSurf,self.winRect)
                                self.is_active = False
                            self.gamestate.move(move)
                            self.drawstone(self.BLACK,x,y)
                        else:
                            print('Illegal move')

                elif event.type == QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()

    #converts mouse coordinates to grid coordinate
    def find_index(self,x,y):
        xcoord = (x-100)//self.grid_size+1
        ycoord = (y-100)//self.grid_size+1
        return xcoord, ycoord



if __name__ == '__main__':
    display = BoardDisplay()
    display.drawboard()
    display.run()
