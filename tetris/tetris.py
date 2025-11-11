import pygame
import random
import sys
import os

# Ensure project root is on sys.path so `from player.player import Grid` works when
# this script is executed directly (e.g. `python tetris/tetris.py`). Without
# this, Python may not resolve the `player` package when the current working
# directory differs from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from player.player import Grid

pygame.init()

height, width = 500, 300
cell = 20
rows = (height - 120) // cell
cols = width // cell

screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
pygame.display.set_caption("Auto-cognito")

black = (0, 0, 0)
white = (255, 255, 255)
lose = (252, 91, 122)
win = (50, 230, 50)
bg_color = (31, 25, 76)
grid_color = (100, 100, 255)

assets_path = os.path.join(os.path.dirname(__file__), "assets")

assets = {
    1: pygame.image.load(os.path.join(assets_path, "1.png")).convert_alpha(),
    2: pygame.image.load(os.path.join(assets_path, "2.png")).convert_alpha(),
    3: pygame.image.load(os.path.join(assets_path, "3.png")).convert_alpha(),
    4: pygame.image.load(os.path.join(assets_path, "4.png")).convert_alpha(),
}

font = pygame.font.SysFont("verdana", 50)
font_2 = pygame.font.SysFont("verdana", 15)


class shape:
    version = {
        "I": [[1, 5, 9, 13], [4, 5, 6, 7]],
        "Z": [[4, 5, 9, 10], [2, 6, 5, 9]],
        "S": [[6, 7, 9, 10], [1, 5, 6, 10]],
        "L": [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        "J": [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        "T": [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        "O": [[1, 2, 5, 6]],
    }
    shapes = ["I", "Z", "S", "L", "J", "T", "O"]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.choice(self.shapes)
        self.shape = self.version[self.type]
        self.color = random.randint(1, 4)
        self.rotation = 0

    def img(self):
        return self.shape[self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.shape)


class tetris:
    def __init__(self, rows, cols):
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.current_shape = None
        self.rows = rows
        self.cols = cols
        self.lvl = 1
        self.next = None
        self.end = False
        self.score = 0
        self.new_shape()

    def make_grid(self):
        for i in range(self.rows + 1):
            pygame.draw.line(screen, grid_color, (0, cell * i), (width, cell * i))
        for i in range(self.cols + 1):
            pygame.draw.line(
                screen, grid_color, (cell * i, 0), (cell * i, height - 120)
            )

    def new_shape(self):
        if not self.next:
            self.next = shape(5, 0)
        self.fig = self.next
        self.next = shape(5, 0)

    def collision(self) -> bool:
        if not self.fig:
            return False
        for i in range(4):
            for j in range(4):
                if (i * 4 + j) in self.fig.img():
                    block_row = i + self.fig.y
                    block_col = j + self.fig.x
                    if (
                        block_row >= self.rows
                        or block_row < 0
                        or block_col < 0
                        or block_col >= self.cols
                    ):
                        return True
                    if block_row >= 0 and self.grid[block_row][block_col] > 0:
                        return True
        return False

    def remove_row(self):
        rerun = False
        for i in range(self.rows - 1, 0, -1):
            completed = True
            for j in range(0, self.cols):
                if self.grid[i][j] == 0:
                    completed = False

            if completed:
                del self.grid[i]
                self.grid.insert(0, [0 for i in range(self.cols)])
                self.score += 1

                if self.score % 5 == 0:
                    self.lvl += 1
                rerun = True

        if rerun:
            self.remove_row()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if (i * 4 + j) in self.fig.img():
                    self.grid[self.fig.y + i][self.fig.x + j] = self.fig.color

        self.remove_row()
        self.new_shape()
        if self.collision():
            self.end = True

    def move(self):
        self.fig.y += 1
        if self.collision():
            self.fig.y -= 1
            self.freeze()

    def left(self):
        self.fig.x -= 1
        if self.collision():
            self.fig.x += 1

    def right(self):
        self.fig.x += 1
        if self.collision():
            self.fig.x -= 1

    def freefall(self):
        while not self.collision():
            self.fig.y += 1
        self.fig.y -= 1
        self.freeze()

    def fast_drop(self):
        for _ in range(3):
            self.fig.y += 1
            if self.collision():
                self.fig.y -= 1
                break

    def rotate(self):
        old_rotation = self.fig.rotation
        self.fig.rotate()
        if self.collision():
            self.fig.rotation = old_rotation
            
    def end_game(self):
        popup = pygame.Rect(50, 140, width - 100, height - 350)
        pygame.draw.rect(screen, black, popup)
        pygame.draw.rect(screen, lose, popup, 2)

        game_over = font_2.render("GAME OVER!", True, white)
        option1 = font_2.render("Press r to restart", True, lose)
        option2 = font_2.render("Press q to quit", True, lose)

        screen.blit(game_over, (popup.centerx - game_over.get_width() / 2, popup.y + 20))
        screen.blit(option1, (popup.centerx - option1.get_width() / 2, popup.y + 60))
        screen.blit(option2, (popup.centerx - option2.get_width() / 2, popup.y + 100))

def dev_main():
    from player.player import update_game_state
    
    run = True
    game = tetris(rows, cols)
    player_grid = Grid()
    cnt = 0
    move = True
    space_press = False
    
    update_game_state(game)
    
    last_keys = {pygame.K_LEFT: False, pygame.K_RIGHT: False, 
                 pygame.K_DOWN: False, pygame.K_UP: False, pygame.K_SPACE: False}
    
    while run:
        screen.fill(bg_color)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                sys.exit()
        keys = pygame.key.get_pressed()
        if not game.end:
            if keys[pygame.K_LEFT] and not last_keys[pygame.K_LEFT]:
                game.left()
            elif keys[pygame.K_RIGHT] and not last_keys[pygame.K_RIGHT]:
                game.right()
            elif keys[pygame.K_DOWN] and not last_keys[pygame.K_DOWN]:
                game.fast_drop()
            elif keys[pygame.K_UP] and not last_keys[pygame.K_UP]:
                game.rotate()
            elif keys[pygame.K_SPACE] and not last_keys[pygame.K_SPACE]:
                space_press = True
        
        last_keys[pygame.K_LEFT] = keys[pygame.K_LEFT]
        last_keys[pygame.K_RIGHT] = keys[pygame.K_RIGHT]
        last_keys[pygame.K_DOWN] = keys[pygame.K_DOWN]
        last_keys[pygame.K_UP] = keys[pygame.K_UP]
        last_keys[pygame.K_SPACE] = keys[pygame.K_SPACE]
                
        if keys[pygame.K_r] and game.end:
            game.__init__(rows, cols)

        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            run = False
            sys.exit()

        cnt += 1
        if cnt >= 1000:
            cnt = 0

        if move:
            if (cnt % (15 // game.lvl * 1.5)) == 0:
                if not game.end:
                    if space_press:
                        game.freefall()
                        space_press = False
                    else:
                        game.move()
        game.make_grid()

        for x in range(rows):
            for y in range(cols):
                if game.grid[x][y] > 0:
                    val = game.grid[x][y]
                    img = assets[val]
                    screen.blit(img, (y * cell, x * cell))
                    pygame.draw.rect(screen, white, (y * cell, x * cell, cell, cell), 1)
        if game.fig:
            for i in range(4):
                for j in range(4):
                    if (i * 4 + j) in game.fig.img():
                        x = (game.fig.x + j) * cell
                        y = (game.fig.y + i) * cell
                        img = pygame.transform.scale(
                            assets[game.fig.color], (cell - 2, cell - 2)
                        )
                        screen.blit(img, (x + 1, y + 1))

        if game.next:
            for i in range(4):
                for j in range(4):
                    if (i * 4 + j) in game.next.img():
                        img = assets[game.next.color]
                        x = (game.next.x + j - 4) * cell
                        y = (game.next.y + i) * cell + height - 100
                        screen.blit(img, (x, y))
                        
        if game.end:
            game.end_game()

        score_txt = font.render(f"{game.score}", True, white)
        lvl_txt = font_2.render(f"Level: {game.lvl}", True, white)
        screen.blit(score_txt, (250 - score_txt.get_width() // 2, height - 120))
        screen.blit(lvl_txt, (250 - score_txt.get_width() // 2, height - 30))

        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    dev_main()
