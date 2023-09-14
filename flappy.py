import pygame
import os
import time
import random
import neat

pygame.font.init()

WIDTH, HEIGHT = 600, 800

GEN = -1

clock = pygame.time.Clock()

FPS = 60

pygame.display.set_caption("Flappy Bird AI")

FLOOR_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "base.png")))
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "pipe.png")))
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "bird3.png")))]

STAT_FONT = pygame.font.SysFont("arial", 50)
GEN_FONT = pygame.font.SysFont("arial", 35)
ALIVE_FONT = pygame.font.SysFont("arial", 35)
pygame.display.set_icon(BIRD_IMGS[0])

# Background
BG = pygame.transform.scale(pygame.image.load(os.path.join("assets", "bg.png")), (WIDTH, HEIGHT))

WIN = pygame.display.set_mode((WIDTH, HEIGHT))



class Base:
    def __init__(self, x, y, vel):
        self.x = x
        self.y = y
        self.vel = vel

class Floor(Base):
    def __init__(self, x, y, vel, img):
        super().__init__(x, y, vel)
        self.img = img
    
    def draw(self):
        WIN.blit(self.img, (self.x, self.y))

        self.x -= self.vel

        if self.x <= -50:
            self.x = 0


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 12.5
    ROT_VEL = 10
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y
    
    def move(self):
        self.tick_count += 1

        d = self.vel * self.tick_count + 1.5 * self.tick_count**2

        if d >= 16:
            d = 16
        
        if d < 0:
            d -= 2
        
        self.y = self.y + d/2

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
    
    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2
    
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 2.5

    def __init__(self, x):
        self.x = x
        self.height = 0
        
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()
    
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
    
    def move(self):
        self.x -= self.VEL
    
    def draw(self):
        WIN.blit(self.PIPE_TOP, (self.x, self.top))
        WIN.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False



def main(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)


    run = True
    score = 0

    # Floor
    floor_vel = 2.5
    floor_x = 0
    floor_y = HEIGHT - FLOOR_IMG.get_height()/2
    floor_img = FLOOR_IMG
    floor = Floor(floor_x, floor_y, floor_vel, floor_img)
    
    pipes = [Pipe(700)]

    def redraw_window(pipes):

        WIN.fill((0, 0, 0))
        
        WIN.blit(BG, (0, 0))

        for pipe in pipes:
            pipe.draw()

        floor.draw()
        for bird in birds:
            bird.draw(WIN)

        text = STAT_FONT.render(f"Score: {str(score)}", 16, (255, 255, 255))
        WIN.blit(text, (WIDTH - 10 - text.get_width(), 10))

        gen = GEN_FONT.render(f"Gen: {str(GEN)}", 16, (255, 255, 255))
        WIN.blit(gen, (10, 10))
        
        alive = ALIVE_FONT.render(f"Birds Alive: {str(len(birds))}", 16, (255, 255, 255))
        WIN.blit(alive, (10, 20 + gen.get_height()))
        pygame.display.update()
    
    while run:
        clock.tick(FPS)

        redraw_window(pipes)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break
        
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            
            if output[0] > 0.5:
                bird.jump()
            
            
        add_pipe = False
        remove = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
            
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
                
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                remove.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(700))
        
        for r in remove:
            pipes.remove(r)
        
        for x, bird in enumerate(birds):
            if bird.y < 0 or bird.y > floor_y - 10:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)
    

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)