#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An attempt at some simple, self-contained pygame-based examples.
Example 01
In short:
One static body: a big polygon to represent the ground
One dynamic body: a rotated big polygon
And some drawing code to get you going.
kne
"""
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, staticBody, circleShape, fixtureDef,dynamicBody,revoluteJointDef,contactListener)
pygame.font.init()
BULLET_FONT = pygame.font.SysFont('arial',20)

import math
import numpy as np
from gym.utils import seeding
import random
import neat
import time


#from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 700

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Simple pygame example')
clock = pygame.time.Clock()


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class World_Starship():
    def __init__(self,LANDING_PAD):
        self.reset(LANDING_PAD)
    def reset(self,LANDING_PAD ):

        self.MAIN_ENGINE_POWER = 7.0
        self.SIDE_ENGINE_POWER = 0.5
        self.FUEL = 0
        self.game_over = False
        self.alive = True
        self.just_dead = True



        self.scal = 3
        self.LANDER_POLY =[
            (-4.5*self.scal, -20*self.scal), (+4.5*self.scal, -20*self.scal), (+4.5*self.scal ,+20*self.scal),
            (0*self.scal, 30*self.scal), (-4.5*self.scal, +20*self.scal)
            ]

        self.LEG_AWAY = 1
        self.LEG_DOWN = 80
        self.LEG_W, self.LEG_H = 2, 20
        self.LEG_SPRING_TORQUE = 400
        self.LEG_ANGLE = 0.9
        self.LEG_ANGLE_MAX = 0.5

        

        self.VIEWPORT_W = SCREEN_WIDTH
        self.VIEWPORT_H = SCREEN_HEIGHT




        self.LANDING_PAD = LANDING_PAD
        



        # --- pygame setup ---
        

        # --- pybox2d world setup ---
        # Create the world
        #world = Box2D.b2World()
        self.world = world(gravity=(0, -8), doSleep=True)#-10
        self.ground_body = self.world.CreateStaticBody(
            position=(0, 1),
            shapes=polygonShape(box=(100, 1)),
        )

        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref




        self.SCALE = 75.0


        self.initial_y = self.VIEWPORT_H/PPM
        self.lander = self.world.CreateDynamicBody(
                    position=(self.VIEWPORT_W/PPM*0.8, self.initial_y),
                    angle=0.0,
                    fixtures = fixtureDef(
                        shape=polygonShape(vertices=[(x/self.SCALE, y/self.SCALE) for x, y in self.LANDER_POLY]),
                        density=5.0,
                        friction=0.1,
                        categoryBits=0x0010,
                        maskBits=0x001,   # collide only with ground
                        restitution=0.0)  # 0.99 bouncy
                        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)


        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                        position=(self.VIEWPORT_W/self.SCALE/2 - i*self.LEG_AWAY/self.SCALE, self.initial_y),
                        angle=(i * 0.05),
                        fixtures=fixtureDef(
                            shape=polygonShape(box=(self.LEG_W/self.SCALE, self.LEG_H/self.SCALE)),
                            density=1.0,
                            restitution=0.0,
                            categoryBits=0x0020,
                            maskBits=0x001)
                        )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                        bodyA=self.lander,
                        bodyB=leg,
                        localAnchorA=(0, 0),
                        localAnchorB=(i * self.LEG_AWAY/self.SCALE, self.LEG_DOWN/self.SCALE),
                        enableMotor=True,
                        enableLimit=True,
                        maxMotorTorque=self.LEG_SPRING_TORQUE,
                        motorSpeed=+0.3 * i  # low enough not to jump back into the sky
                        )
            if i == -1:
                rjd.lowerAngle = +self.LEG_ANGLE - self.LEG_ANGLE_MAX  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +self.LEG_ANGLE
            else:
                rjd.lowerAngle = -self.LEG_ANGLE
                rjd.upperAngle = -self.LEG_ANGLE + self.LEG_ANGLE_MAX
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs


        self.particles = []

        self.world.Step(TIME_STEP, 10, 10)
        self.lander.angle = math.radians(90)
        self.legs[0].angle = math.radians(90)
        self.legs[1].angle = math.radians(90)



        self.colors = {
        staticBody: (255, 255, 255, 255),
        dynamicBody: (127, 127, 127, 255),
        leg: (255,0,0,255),
        self.lander:(122, 255, 24,255)

        }

    def distance_to_pad(self):
        ss = SCREEN_HEIGHT-40

        pos = list(self.lander.position)
        pos[0] = pos[0]*PPM
        pos[1] = SCREEN_HEIGHT- pos[1]*PPM
        LANDING_PAD = self.LANDING_PAD
        self.DISTANCE_TO_PAD = [math.sqrt((pos[0]-LANDING_PAD[0])**2+(pos[1]-ss)**2),
                        math.sqrt((pos[0]-(LANDING_PAD[1]+LANDING_PAD[0]))**2+(pos[1]-ss)**2)]
        return self.DISTANCE_TO_PAD
    def get_data(self):
        data = [
        #self.lander.position[0],
        #self.lander.position[1],
        self.lander.linearVelocity[0],
        self.lander.linearVelocity[1],
        self.lander.angle,
        self.lander.angularVelocity,
        1#self.FUEL

        ] + self.distance_to_pad()
        return data





    def _create_particle(self, mass, x, y, ttl):
            p = self.world.CreateDynamicBody(
                position = (x, y),
                angle=0.0,
                fixtures = fixtureDef(
                    shape=circleShape(radius=2/self.SCALE, pos=(0, 0)),
                    density=mass,
                    friction=0.1,
                    categoryBits=0x0100,
                    maskBits=0x001,  # collide only with ground
                    restitution=0.3)
                    )
            p.ttl = ttl
            self.particles.append(p)
            #self._clean_particles(False)
            return p

    def _clean_particles(self, all):
        while particles and (all or particles[0].ttl < 0):
            self.world.DestroyBody(particles.pop(0))


    def motor_up(self,p,d=None):

            tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
            if d!= None:
                r = math.radians((30*d))
                tip  = (math.sin(self.lander.angle+r), math.cos(self.lander.angle+r))

            side = (-tip[1], tip[0])
            np_random, seed = seeding.np_random()
            dispersion = [np_random.uniform(-0.5, +0.5) / self.SCALE for _ in range(2)]
            #dispersion = [np_random.uniform(-0.1, +0.1) / SCALE for _ in range(2)]

            m_power = 2.0
            v_m_power = m_power*0.3
            ox = (tip[0] * (4/self.SCALE + 2 * dispersion[0]) + side[0] * dispersion[1])  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4/self.SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1]-2 + oy)


            MAIN_ENGINE_HEIGHT = -60
            MAIN_ENGINE_AWAY = 0

            qx =   tip[1]* MAIN_ENGINE_AWAY/self.SCALE    - tip[0] * MAIN_ENGINE_HEIGHT/self.SCALE 
            qy =   tip[0]* MAIN_ENGINE_AWAY/self.SCALE    + tip[1] * MAIN_ENGINE_HEIGHT/self.SCALE 

            impulse_pos2 = (self.lander.position[0] +  qx ,
                               self.lander.position[1] +  qy    )


            p = self._create_particle(0.7,( impulse_pos2[0]), (impulse_pos2[1]), v_m_power)
            
            p.ApplyLinearImpulse((ox * self.MAIN_ENGINE_POWER * v_m_power, oy * self.MAIN_ENGINE_POWER * v_m_power),impulse_pos2, True)

            
            MAIN_ENGINE_HEIGHT = 60
            MAIN_ENGINE_AWAY = 0

            mx =   tip[1]* MAIN_ENGINE_AWAY/self.SCALE    - tip[0] * MAIN_ENGINE_HEIGHT/self.SCALE 
            my =   tip[0]* MAIN_ENGINE_AWAY/self.SCALE    + tip[1] * MAIN_ENGINE_HEIGHT/self.SCALE 


            self.lander.ApplyLinearImpulse((-ox * self.MAIN_ENGINE_POWER * m_power, -oy * self.MAIN_ENGINE_POWER * m_power),
                                               impulse_pos2,
                                               True)

            """
            poss = list(impulse_pos)
            poss[0] = poss[0]*PPM
            poss[1] = SCREEN_HEIGHT- poss[1]*PPM
            pygame.draw.circle(screen, (255,0,0),poss,2)
            #print('poss1',poss)

            poss = list(impulse_pos2)
            poss[0] = poss[0]*PPM
            poss[1] = SCREEN_HEIGHT- poss[1]*PPM
            pygame.draw.circle(screen, (255,255,0),poss,5)

            poss = [self.lander.position[0]+ mx,self.lander.position[1]+my]
            poss[0] = poss[0]*PPM
            poss[1] = SCREEN_HEIGHT- poss[1]*PPM
            pygame.draw.circle(screen, (255,100,0),poss,5)

            poss = list(self.lander.position)
            poss[0] = poss[0]*PPM
            poss[1] = SCREEN_HEIGHT- poss[1]*PPM
            pygame.draw.circle(screen, (255,0,255),poss,10)

            poss = [-ox * self.MAIN_ENGINE_POWER * m_power, -oy * self.MAIN_ENGINE_POWER * m_power]
            poss[0] += impulse_pos[0]
            poss[1] += impulse_pos[1]
            poss[0] = poss[0]*PPM
            poss[1] = SCREEN_HEIGHT- poss[1]*PPM
            pygame.draw.circle(screen, (0,0,255),poss,10)
            """


            
            self.FUEL += 0.8 
            





    def motor_side(self,p,d):
                direction = -d   
                s_power = p
                v_s_power = s_power*0.5
                SIDE_ENGINE_HEIGHT = 50.0
                SIDE_ENGINE_AWAY = 10.0


                tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
                side = (-tip[1], tip[0])
                np_random, seed = seeding.np_random()
                dispersion = [np_random.uniform(-1.0, +1.0) / self.SCALE for _ in range(2)]
                #tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
                ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/self.SCALE)
                oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/self.SCALE)


                impulse_pos = (self.lander.position[0] + ox - tip[0] * SIDE_ENGINE_AWAY/self.SCALE,
                               self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT/self.SCALE)

                
                qx =   tip[1] * SIDE_ENGINE_AWAY/self.SCALE  - tip[0] * SIDE_ENGINE_HEIGHT/self.SCALE 
                qy =   tip[0] *SIDE_ENGINE_AWAY/self.SCALE  + tip[1] * SIDE_ENGINE_HEIGHT/self.SCALE 

                impulse_pos2 = (self.lander.position[0] +  qx ,
                               self.lander.position[1] +  qy    )

               
                

                p = self._create_particle(0.7, impulse_pos2[0], impulse_pos2[1], v_s_power)
                
                p.ApplyLinearImpulse((ox * self.SIDE_ENGINE_POWER * v_s_power, oy * self.SIDE_ENGINE_POWER * v_s_power),
                                     impulse_pos2
                                     , True)
                
                
                self.lander.ApplyLinearImpulse((-ox * self.SIDE_ENGINE_POWER * s_power, -oy * self.SIDE_ENGINE_POWER * s_power),impulse_pos,True)
                #print((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),impulse_pos,True)
                #print('impuls ',impulse_pos2, p.position)
                """
                poss = list(impulse_pos2)
                poss[0] = poss[0]*PPM
                poss[1] = SCREEN_HEIGHT- poss[1]*PPM
                pygame.draw.circle(screen, (255,0,0),poss,5)


                poss = [-ox * self.SIDE_ENGINE_POWER * s_power, -oy * self.SIDE_ENGINE_POWER * s_power]
                poss[0] += impulse_pos[0]
                poss[1] += impulse_pos[1]
                poss[0] = poss[0]*PPM
                poss[1] = SCREEN_HEIGHT- poss[1]*PPM
                pygame.draw.circle(screen, (0,0,255),poss,10)
                """

                
                self.FUEL += 0.3 






# --- main game loop ---
def main(genomes, config):
    nets = []
    fit = []
    ships = []

    LANDING_PAD_LENGHT = SCREEN_WIDTH*0.2
    LANDING_PAD_MARRGIN = SCREEN_WIDTH*0.1
    LANDING_PAD_MARRGIN_LEFT = SCREEN_WIDTH*0.3

    LANDING_PAD = [random.randint(LANDING_PAD_MARRGIN,SCREEN_WIDTH-LANDING_PAD_MARRGIN_LEFT-LANDING_PAD_LENGHT)]
    LANDING_PAD.append(LANDING_PAD_LENGHT)
    
    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ships.append(World_Starship(LANDING_PAD))
    running = True
    K_UP =False
    K_LEFT =False
    K_RIGHT =False
    K_a =False
    K_e =False




    world = ships[0].world
    ground_body =  ships[0].ground_body
    drawlist =  ships[0].drawlist
    colors = ships[0].colors
    lander = ships[0].lander

    alive = 1

    while running:
        # Check the event queue
        screen.fill((0, 0, 0, 0))
        if alive <= 0:
            #time.sleep(3)
            """
            for i in genomes:

                print(i[1].fitness,max_gen_fit)
            """
            break  



        particles = []
        for i in ships:
            particles = particles + i.particles


        part_ = []

        for obj in particles:
                
                #for obj in self.particles:
                obj.ttl -= 0.15
                mul = max(1/max(0.2, 0.5+obj.ttl),1/max(0.2, 0.2+obj.ttl))

                col1 = (1-(max(0.2, 0.2+obj.ttl)*mul))*255
                col2 = (1-(max(0.2, 0.5*obj.ttl)*mul))*255
                
                obj.color1 = col2, col1, col1
                obj.color2 = col2,col1, col1
                if obj.ttl >0.0:
                    part_.append(obj)
        particles = part_




       # print(drawlist)
        tp_do  = [ground_body]+particles #+ drawlist +ships[1].drawlist #
        #print(tp_do[-1])
        for w in ships:
            tp_do.append(w.drawlist[0] )
            tp_do.append(w.drawlist[1] )
            tp_do.append(w.drawlist[2] )
        #print(tp_do[-1],ships[2].drawlist )

        for body in tp_do :  # or: world.bodies
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                # The fixture holds information like density and friction,
                # and also the shape.
                #print(type(fixture.shape),particles)
                #trans = fixture.body.transform
                if type(fixture.shape) is circleShape:

                    
                    multi = 2
                    #print('draw pos', body.position)
                    
                    tmp_pos = list(body.position)
                    #tmp_pos[0] = tmp_pos[0]+lander.position[0]*PPM
                    #tmp_pos[1] = SCREEN_HEIGHT-(tmp_pos[1]+ lander.position[1]*PPM)
                    tmp_pos[0] = tmp_pos[0]*PPM
                    tmp_pos[1] = SCREEN_HEIGHT-(tmp_pos[1]*PPM)
                    #print(list(body.position),tmp_pos)
                    #print(lander.position)

                    pygame.draw.circle(screen,body.color2 ,tmp_pos,(fixture.shape.radius+2)*multi)
                    pygame.draw.circle(screen, body.color1,tmp_pos,fixture.shape.radius*multi)
                    #print(body.position,fixture.shape.radius*multi)





                else:






                    shape = fixture.shape

                    # Naively assume that this is a polygon shape. (not good normally!)
                    # We take the body's transform and multiply it with each
                    # vertex, and then convert from meters to pixels with the scale
                    # factor.
                    vertices = [(body.transform * v) * PPM for v in shape.vertices]

                    # But wait! It's upside-down! Pygame and Box2D orient their
                    # axes in different ways. Box2D is just like how you learned
                    # in high school, with positive x and y directions going
                    # right and up. Pygame, on the other hand, increases in the
                    # right and downward directions. This means we must flip
                    # the y components.
                    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
                    pygame.draw.polygon(screen, colors[body.type], vertices)


        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                running = False
            if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z :
                        K_UP = True
                    if event.key == pygame.K_q :
                        K_LEFT = True
                    if event.key == pygame.K_d :
                        K_RIGHT = True
                    if event.key == pygame.K_a :
                        K_a = True
                    if event.key == pygame.K_e :
                        K_e = True
                        
                        
            if event.type == pygame.KEYUP:
                    if event.key == pygame.K_z :
                        K_UP = False
                    if event.key == pygame.K_q :
                        K_LEFT = False
                    if event.key == pygame.K_d :
                        K_RIGHT = False
                    if event.key == pygame.K_a :
                        K_a = False
                    if event.key == pygame.K_e :
                        K_e = False
        if K_UP == True:
            for w in ships:
                w.motor_up(1)
        if K_LEFT == True:
            for w in ships:
                w.motor_up(1,-1)
        if K_RIGHT == True:
            for w in ships:
                w.motor_up(1,1)

        if K_a == True:
            for w in ships:
                w.motor_side(1,1)
        if K_e == True:
            for w in ships:
                w.motor_side(1,-1)




        max_gen_fit = genomes[0][1].fitness

        for i in genomes:
            if (i[1].fitness)>max_gen_fit:
                max_gen_fit = i[1].fitness
        

        # Draw the world
        alive =0
        for index, ship in enumerate(ships):

            pos = list(ship.lander.position)
            pos[0] = pos[0]*PPM
            pos[1] = SCREEN_HEIGHT- pos[1]*PPM
           
            
            if pos[0]> SCREEN_WIDTH:
                #pos[0] = SCREEN_WIDTH
                ship.game_over = True
            if pos[1]> SCREEN_HEIGHT:
                #pos[1] = SCREEN_HEIGHT
                ship.game_over = True
            if pos[0]< -2:
                #pos[0] = 0
                ship.game_over = True
            if pos[1]< -2:
                #pos[1] = 0
                ship.game_over = True

            try:
                #pos[1] -= 70
                #pos[0] += 10
                txt = genomes[index][1].fitness
                txt= round((ship.distance_to_pad()[0]+ship.distance_to_pad()[1])/2,0)
                txt = ship.lander.angle
                
                if round(genomes[index][1].fitness,1) == round(max_gen_fit,1) :
                    pass

                   # DISTANCE_TO_PADD = BULLET_FONT.render(str(txt), False, (255, 0, 255))
                    #pygame.draw.circle(screen, (0,0,255), ((LANDING_PAD[0]+LANDING_PAD[1])/2, 50), txt, 50)
                    #pygame.draw.circle(screen, (0,0,255), pos, txt,1)
             


                screen.blit(DISTANCE_TO_PADD,pos)
            except:
                pass
            

            
            if ship.alive == True:
                alive += 1
                output = nets[index].activate(ship.get_data())
                s_t = output[0]
                t_d = output[1]
                m_o = output[2]

                if m_o >0.5:
                    if t_d> 0.5:
                        ship.motor_up(1,1)
                    elif t_d< 0.5:
                        ship.motor_up(1,-1)
                    else:
                        ship.motor_up(1)
                if s_t> 0.5:
                    ship.motor_side(1,1)
                elif s_t> -0.5:
                    ship.motor_side(1,-1)


                #genomes[index][1].fitness += 1



                if ship.game_over == True and ship.just_dead == True:
                    
                    ship.just_dead = False
                    ship.alive = False
                    #print(genomes[index][1].fitness,'here')
                    genomes[index][1].fitness -= 100
                    #genomes[index][1].fitness -= ship.FUEL
                    genomes[index][1].fitness -= (ship.distance_to_pad()[0]+ship.distance_to_pad()[1])
                    if ship.lander.angle>0:
                        genomes[index][1].fitness -= ship.lander.angle*20
                    else:
                        genomes[index][1].fitness += ship.lander.angle*20


                    if ship.lander.angularVelocity>0:
                        genomes[index][1].fitness -= ship.lander.angularVelocity*20
                    else:
                        genomes[index][1].fitness += ship.lander.angularVelocity*20

                    if ship.lander.linearVelocity[0] + ship.lander.linearVelocity[1] >0:
                        genomes[index][1].fitness -= (ship.lander.linearVelocity[0] + ship.lander.linearVelocity[1])*10
                    else:
                        genomes[index][1].fitness += (ship.lander.linearVelocity[0] + ship.lander.linearVelocity[1])*10
                    

                    #genomes[index][1].fitness += ship.FUEL-500
                    print(ship.get_data(),'here',(ship.distance_to_pad()[0]+ship.distance_to_pad()[1]),ship.lander.angle*20,ship.lander.angularVelocity*20,(ship.lander.linearVelocity[0] + ship.lander.linearVelocity[1])*10,genomes[index][1].fitness)
                if ship.lander.awake == False and ship.just_dead == True:
                    print('game awake')
                    ship.just_dead = False
                    ship.alive = False
                    genomes[index][1].fitness += 200
                    genomes[index][1].fitness -= (ship.distance_to_pad()[0]+ship.distance_to_pad()[1])*0.1
                    if ship.lander.angle>0:
                        genomes[index][1].fitness -= ship.lander.angle*2
                    else:
                        genomes[index][1].fitness += ship.lander.angle*2
                    if ship.lander.linearVelocity[0] + ship.lander.linearVelocity[1] >0:
                        genomes[index][1].fitness -= ship.lander.linearVelocity[0] + ship.lander.linearVelocity[1]
                    else:
                        genomes[index][1].fitness += ship.lander.linearVelocity[0] + ship.lander.linearVelocity[1]
                    #genomes[index][1].fitness += ship.FUEL-500


                    #print(ship.get_data(),'here',(ship.game_over,ship.lander.awake))

                ship.world.Step(TIME_STEP, 10, 10)

            
        

        ss = SCREEN_HEIGHT-40
        x,y = LANDING_PAD
        
        #print(LANDING_PAD,LANDING_PAD[1]-LANDING_PAD[0] )
        pygame.draw.rect(screen,(50,50,50), (x,ss,y, 10))

        '''
        state = [
                (pos.x - SCREEN_WIDTH/SCALE/2) / (SCREEN_WIDTH/SCALE/2),
                (pos.y - (helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
                vel.x*(VIEWPORT_W/SCALE/2)/FPS,
                vel.y*(VIEWPORT_H/SCALE/2)/FPS,
                self.lander.angle,
                20.0*self.lander.angularVelocity/FPS,
                1.0 if self.legs[0].ground_contact else 0.0,
                1.0 if self.legs[1].ground_contact else 0.0
                ]
        '''
     
       
        # Flip the screen and try to keep at the target FPS

        pygame.display.flip()
        clock.tick(TARGET_FPS)


if __name__ == "__main__":
    config_path = "config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Run NEAT
    p.run(main, 1000)



pygame.quit()
print('Done!')