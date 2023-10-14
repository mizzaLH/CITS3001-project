from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string

# code for locating objects on the screen in super mario bros
# by Lauren Gee

# Template matching is based on this tutorial:
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

################################################################################

# change these values if you want more/less printing
PRINT_GRID      = False
PRINT_LOCATIONS = True

# If printing the grid doesn't display in an understandable way, change the
# settings of your terminal (or anaconda prompt) to have a smaller font size,
# so that everything fits on the screen. Also, use a large terminal window /
# whole screen.

# other constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# You can add more images to improve the object locator, so that it can locate
# more things. For best results, paint around the object with the exact shade of
# blue as the sky colour. (see the given images as examples)
#
# Put your image filenames in image_files below, following the same format, and
# it should work fine.

# filenames for object templates
image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
    },
    "block": {
        "block1": ["block1.png"],
        "block2": ["block2.png"],
        "block3": ["block3.png"],
        "block4": ["block4.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.

        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    }
}

def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}

# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# PRINTING THE GRID (for debug purposes)

colour_map = {
    (104, 136, 252): " ", # sky blue colour
    (0,     0,   0): " ", # black
    (252, 252, 252): "'", # white / cloud colour
    (248,  56,   0): "M", # red / mario colour
    (228,  92,  16): "%", # brown enemy / block colour
}
unused_letters = sorted(set(string.ascii_uppercase) - set(colour_map.values()),reverse=True)
DEFAULT_LETTER = "?"

def _get_colour(colour): # colour must be 3 ints
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]
    
    # if we haven't seen this colour before, pick a letter to represent it
    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER

def print_grid(obs, object_locations):
    pixels = {}
    # build the outlines of located objects
    for category in object_locations:
        for location, dimensions, object_name in object_locations[category]:
            x, y = location
            width, height = dimensions
            name_str = object_name.replace("_", "-") + "-"
            for i in range(width):
                pixels[(x+i, y)] = name_str[i%len(name_str)]
                pixels[(x+i, y+height-1)] = name_str[(i+height-1)%len(name_str)]
            for i in range(1, height-1):
                pixels[(x, y+i)] = name_str[i%len(name_str)]
                pixels[(x+width-1, y+i)] = name_str[(i+width-1)%len(name_str)]

    # print the screen to terminal
    print("-"*SCREEN_WIDTH)
    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                # this pixel is part of an outline of an object,
                # so use that instead of the normal colour symbol
                colour = pixels[coords]
            else:
                # get the colour symbol for this colour
                colour = _get_colour(obs[y][x])
            line.append(colour)
        print("".join(line))

################################################################################
# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break
    
    #      [((x,y), (width,height))]
    return [( loc,  locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))
    
    # stop early if there are no pipes
    if not upper_locs:
        return []
    
    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y+h, x+2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations


def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue
            
            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # initialize the "pipe" category as an empty list
    object_locations["pipe"] = []  # 初始化 "pipe" 类别为一个空列表
    # locate pipes
    object_locations["pipe"] += _locate_pipe(screen)

    return object_locations

################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

def make_action(screen, info, step, env, prev_action):
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    # You probably don't want to print everything I am printing when you run
    # your code, because printing slows things down, and it puts a LOT of
    # information in your terminal.

    # Printing the whole grid is slow, so I am only printing it occasionally,
    # and I'm only printing it for debug purposes, to see if I'm locating objects
    # correctly.
    if PRINT_GRID and step % 100 == 0:
        print_grid(screen, object_locations)
        # If printing the grid doesn't display in an understandable way, change
        # the settings of your terminal (or anaconda prompt) to have a smaller
        # font size, so that everything fits on the screen. Also, use a large
        # terminal window / whole screen.

        # object_locations contains the locations of all the objects we found
        print(object_locations)

    # List of locations of Mario:
    mario_locations = object_locations["mario"]
    # (There's usually 1 item in mario_locations, but there could be 0 if we
    # couldn't find Mario. There might even be more than one item in the list,
    # but if that happens they are probably approximately the same location.)

    # List of locations of enemies, such as goombas and koopas:
    enemy_locations = object_locations["enemy"]

    # List of locations of blocks, pipes, etc:
    block_locations = object_locations["block"]
    pipe_locations = object_locations["pipe"]

    # List of locations of items: (so far, it only finds mushrooms)
    item_locations = object_locations["item"]


    waiting_on_pipe = False
    # This is the format of the lists of locations:
    # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
    #
    # x_coordinate and y_coordinate are the top left corner of the object
    #
    # For example, the enemy_locations list might look like this:
    # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]
    
    if PRINT_LOCATIONS:
        # To get the information out of a list:
        for enemy in enemy_locations:
            enemy_location, enemy_dimensions, enemy_name = enemy
            x, y = enemy_location
            width, height = enemy_dimensions
            print("enemy:", x, y, width, height, enemy_name)

        # Or you could do it this way:
        print("##############################################node11")  
        for block in block_locations:
            block_x = block[0][0]
            block_y = block[0][1]
            block_width = block[1][0]
            block_height = block[1][1]
            block_name = block[2]
            print(f"{block_name}: {(block_x, block_y)}), {(block_width, block_height)}")
        print("##############################################node14")  
        for pipe in pipe_locations:
            pipe_x = pipe[0][0]
            pipe_y = pipe[0][1]
            pipe_width = pipe[1][0]
            pipe_height = pipe[1][1]
            pipe_name = pipe[2]
            print(f"{pipe_name}: {(pipe_x, pipe_y)}), {(pipe_width, pipe_height)}") 
        # Or you could do it this way:
        print("##############################################node13")
        for item_location, item_dimensions, item_name in item_locations:
            x, y = item_location
            print(item_name, x, y)

        # gym-super-mario-bros also gives us some info that might be useful
        print(info)
        # see https://pypi.org/project/gym-super-mario-bros/ for explanations

        # The x and y coordinates in object_locations are screen coordinates.
        # Top left corner of screen is (0, 0), top right corner is (255, 0).
        # Here's how you can get Mario's screen coordinates:
        if mario_locations:
            location, dimensions, object_name = mario_locations[0]
            mario_x, mario_y = location
            print("Mario's location on screen:",
                  mario_x, mario_y, f"({object_name} mario)")
        
        # The x and y coordinates in info are world coordinates.
        # They tell you where Mario is in the game, not his screen position.
        mario_world_x = info["x_pos"]
        mario_world_y = info["y_pos"]
        # Also, you can get Mario's status (small, tall, fireball) from info too.
        mario_status = info["status"]
        print("Mario's location in world:",
              mario_world_x, mario_world_y, f"({mario_status} mario)")

    # TODO: Write code for a strategy, such as a rule based agent.

    # Choose an action from the list of available actions.
    # For example, action = 0 means do nothing
    #              action = 1 means press 'right' button
    #              action = 2 means press 'right' and 'A' buttons at the same time
     # If there are any detected enemies
    # if enemy_locations:
    #     first_enemy_location = enemy_locations[0][0]  # grabbing the location of the first enemy detected
        
    #     # If Mario is found, use its location. If not, it's safer not to jump.
    #     if mario_locations:
    #         mario_location = mario_locations[0][0]
            
    #         # Check if the first enemy is to the right of Mario
    #         if first_enemy_location[0] > mario_location[0] - 18:
    #             return 2  # This corresponds to the "right jump" action in SIMPLE_MOVEMENT

    if enemy_locations:
        # print("##############################################node16") 
        for enemy in enemy_locations:
            print(enemy_locations)
            enemy_location = enemy[0]
            enemy_x, enemy_y = enemy_location
            if mario_locations:
                mario_location = mario_locations[0][0]
                print("Printing mario location in if enemy_locations",mario_location)
                mario_x, mario_y = mario_location
            # Check if the enemy is in front of Mario and within a certain distance (e.g., 50 pixels)
            if prev_action == 2 or prev_action == 4:

                if enemy_x > mario_x and 15 > (enemy_x - mario_x):
                    print("Currently jumping - trying to move right to land on goomba")
                    print("Action = 6")
                    return 1
                elif enemy_x < mario_x and 15 > (mario_x - enemy_x):
                    print("Currently jumping - trying to move left")
                    print("Action = 6")
                    return 6
            if mario_x < enemy_x and (enemy_x - mario_x) < 50:
                
                # Check if the enemy is a high threat
                if is_high_threat(mario_location,enemy_location):
                    
                    # Check if it's safe to jump over the enemy
                    if safe_to_jump_over(mario_location, enemy_location, enemy_locations) and can_jump_over(screen, mario_location,enemy_location):
                        print("Jumping to the right over an enemy")
                        return 4  # This corresponds to the "jump right" action in SIMPLE_MOVEMENT
                    elif not safe_to_jump_over(mario_location, enemy_location, enemy_locations):
                        print("Jump deemed to be not safe,lets move left",)
                        return 6
                    elif not can_jump_over(screen, mario_location,enemy_location):
                        print("Jumped deemed to not be valid as blocks above")
            '''if enemy[0][0] < mario_x:
                continue
            enemy_x = enemy[0][0]  # grabbing the location of the first enemy detected
        # If Mario is found, use its location. If not, it's safer not to jump.
            if mario_locations:
                mario_location = mario_locations[0][0] 
                # Check if the first enemy is to the right of Mario
                if enemy_x - mario_location[0] < 50 and enemy_x - mario_location[0] > 0:
                    if mario_location[1] > 190:
                        if enemy_x - mario_location[0] < 17:
                            print("wanghuizou")
                            return 9
                        if enemy_x - mario_location[0] > 20:
                            print("wotiaole")
                            return 4 
                    # if (mario_location[1] < 160) and (mario_location[1] > 130):
                    #     if enemy_x - mario_location[0] < 20:
                    #         print("aiming the enemies")
                    #         return 3
                    #     if mario_location[0] - enemy_x < 20:
                    #         print("aiming the enemies")
                    #         return 8
                    
                    # print("wotiaole")
                    # return 2  # This corresponds to the "right jump" action in SIMPLE_MOVEMENT
            '''
    if pipe_locations:
        for pipe in pipe_locations:
            pipe_x = pipe[0][0]
            
            if mario_locations:
                mario_x = mario_locations[0][0][0]  # Mario's x-coordinate
                distance_to_pipe = pipe_x - mario_x
                print("Distance to pipe is", distance_to_pipe)
                if distance_to_pipe >= -40:

                    # Check if Mario is on top of the pipe
                    if is_on_top_of_pipe(mario_locations[0], pipe):
                        print("Mario is on top of the pipe")
                        action = action_from_pipe_top(pipe, enemy_locations)
                        print("The action we are returning is", action)
                        return action

                    # Define the distance 'x' from the pipe where Mario should decide to jump
                    x = 16  # Adjust this value based on your game's physics and behavior

                    # Mario in open space, running to the right
                    if distance_to_pipe > x + 20:
                        return 3  # ['right', 'B']

                    # Too close to the pipe, backtrack to distance x
                    if 12 < distance_to_pipe < x:  # 10 is an arbitrary buffer, adjust as needed
                        return 6  # ['left']

                    # At distance x, decide whether to jump over or land on the pipe
                    if x <= distance_to_pipe <= x + 20:
                        if is_safe_to_jump_over_pipe(pipe, enemy_locations):
                            print("Safe to jump over the pipe at distance", distance_to_pipe)
                            return 4  # ['right', 'A', 'B']
                        else:
                            print("Not safe, attempting to land on the pipe!")
                            return 2  # ['right', 'A']

                    # If Mario is on the pipe, wait until it's safe
                    if mario_x > pipe_x and not is_safe_to_jump_over_pipe(pipe, enemy_locations):
                        print("Waiting on top of the pipe for safety")
                        return 0  # ['NOOP']


    if block_locations:
        for index, block in enumerate(block_locations):
            # 获取当前块的X和Y坐标以及高度
            block_x = block[0][0]
            block_y = block[0][1]
            block_height = block[1][1]
            block_width = block[1][0]
            block_name = block[2]
            if index + 1 < len(block_locations):
                # 获取下一个块的值
                next_block = block_locations[index + 1]
                next_block_x = next_block[0][0]
                next_block_y = next_block[0][1]
                next_block_height = next_block[1][1]
                if (block_y == 224) and (next_block_y == 224):
                    current_x = block_x
                    next_x = next_block_x    
                    # 检查两个块之间的间隔是否大于正常间隔
                    if mario_locations:
                        if (next_x - current_x > 16) and (current_x - mario_locations[0][0][0] < 20):
                            print("jumping over the gap")
                            if (next_x - current_x <= 32):
                                return 4
                            if (next_x - current_x > 32):
                                return 4
            #here should to deal with the block4 
            if block_name == 'block4':
                # print("trying to overcome the block4")
                if mario_locations:
                    # highest_block_y = min(block[0][1] for block in block_locations if block_name == 'block4')  # 查找block4的最高位置
                    # highest_blocks = [block for block in block_locations if block[0][1] == highest_block_y and block_name == 'block4']  # 找到所有最高位置的block4
                    # 只考虑 block4 块的最高位置
                    highest_block_y = min(block[0][1] for block in block_locations if block[2] == 'block4')  
                    # 找到所有最高位置的 block4 块
                    highest_blocks = [block for block in block_locations if block[0][1] == highest_block_y and block[2] == 'block4']  
                    print(highest_block_y)
                    # print(highest_blocks)
                    print(highest_blocks[0][0][0])
                    if (mario_locations[0][0][0] <= highest_blocks[0][0][0]):
                        if (mario_locations[0][0][1]-block_y >= -5)and (mario_locations[0][0][1]-block_y <= 5):
                            if (mario_locations[0][0][0] <= block_x) and (block_x - mario_locations[0][0][0] >= 11) and (block_x - mario_locations[0][0][0] <= 13):
                                print("going back a little")
                                return 6
                            if (mario_locations[0][0][0] <= block_x) and (block_x - mario_locations[0][0][0] < 115):
                                print("jump")
                                return 2
                    if (highest_blocks[0][0][0] -  mario_locations[0][0][0] >= 4) and (highest_blocks[0][0][0] -  mario_locations[0][0][0] < 16):
                        print("fast run")
                        return 3
                    if (highest_blocks[0][0][0] -  mario_locations[0][0][0] >= 0) and (highest_blocks[0][0][0] -  mario_locations[0][0][0] < 4):
                        print("long jump")
                        return 4
                    if (highest_blocks[0][0][0] -  mario_locations[0][0][0] < 0) and (highest_blocks[0][0][0] -  mario_locations[0][0][0] > -100):
                        print("fast run")
                        return 3
            #for the question block
            if block_name == 'question_block':
                if mario_locations:
                    if(mario_locations[0][0][0] < block_x) and (block_x - mario_locations[0][0][0] <= 24) and block_x - mario_locations[0][0][0] >= 16:
                        print("jump to touch the question block")
                        # return 2
                    

    


    # if step % 10 == 0:
    #     # I have no strategy at the moment, so I'll choose a random action.
    #     action = 1
    #     return action
    # else:
    #     # With a random agent, I found that choosing the same random action
    #     # 10 times in a row leads to slightly better performance than choosing
    #     # a new random action every step.
   
    if block_locations:
        for block in block_locations:
            block_name = block[2]
            if block_name == 'block4':
                print("return 0")
                return 0
    if enemy_locations:
         for enemy in enemy_locations:
            enemy_location = enemy[0]
            enemy_x, enemy_y = enemy[0]
            if mario_locations:
                mario_location = mario_locations[0][0]
                print("Printing mario location in if enemy_locations",mario_location)
                mario_x, mario_y = mario_location
            # Check if the enemy is in front of Mario and within a certain distance (e.g., 50 pixels)
                if mario_x < enemy_x and (enemy_x - mario_x) < 120:
                    return 1


    print("return 3")
    return 3
    



################################################################################
#Enemy Helper Functions
def is_high_threat(mario_location, enemy_location, threat_distance=30):
    """
    Determines if an enemy is a high threat based on its proximity to Mario.

    Parameters:
    - mario_location: Tuple containing Mario's (x, y) coordinates.
    - enemy_location: Tuple containing the enemy's (x, y) coordinates.
    - threat_distance: The horizontal distance within which an enemy is considered a high threat.

    Returns:
    - True if the enemy is a high threat. False otherwise.
    """

    mario_x, mario_y = mario_location
    enemy_x, enemy_y = enemy_location

    # Calculate horizontal and vertical distances between Mario and the enemy
    horizontal_distance = abs(mario_x - enemy_x)
    vertical_distance = abs(mario_y - enemy_y)

    # Check if the enemy is within the threat distance horizontally
    # and is approximately at the same vertical level as Mario.
    if horizontal_distance < threat_distance and vertical_distance < 20:
        return True

    return False

def can_land_on(mario_location, enemy_location, enemy_type):
    """
    Determines if Mario can safely land on an enemy to defeat it.

    Parameters:
    - mario_location: Tuple containing Mario's (x, y) coordinates.
    - enemy_location: Tuple containing the enemy's (x, y) coordinates.
    - enemy_type: String indicating the type of the enemy (e.g., "goomba", "koopa").

    Returns:
    - True if Mario can safely land on the enemy. False otherwise.
    """

    mario_x, mario_y = mario_location
    enemy_x, enemy_y = enemy_location

    # Check if the enemy is a type that Mario can land on
    landable_enemies = ["goomba", "koopa"]
    if enemy_type not in landable_enemies:
        return False

    # Calculate horizontal and vertical distances between Mario and the enemy
    horizontal_distance = abs(mario_x - enemy_x)
    vertical_distance = enemy_y - mario_y  # We only care if the enemy is below Mario

    # Check if Mario is directly above the enemy and within a certain horizontal range
    if 0 <= horizontal_distance <= 16 and 0 < vertical_distance < 40:  # 16 is roughly the width of Mario/enemy
        return True

    return False

def safe_to_jump_over(mario_location, enemy_location, all_enemies):
    """
    Determines if it's safe for Mario to jump over an enemy.

    Parameters:
    - mario_location: Tuple containing Mario's (x, y) coordinates.
    - enemy_location: Tuple containing the enemy's (x, y) coordinates.
    - all_enemies: List of all enemy locations on the screen.
    - all_obstacles: List of all obstacle locations on the screen.

    Returns:
    - True if it's safe for Mario to jump over the enemy. False otherwise.
    """
    print("printing mario locations in safe_to_jump_over",mario_location)
    mario_x = mario_location[0]
    print("printing mario_x locations in safe_to_jump_over",mario_x)
    mario_y = mario_location[1]
    enemy_x = enemy_location[0]
    enemy_y = enemy_location[1]

    # Define a landing zone after jumping over the enemy
    # Assuming Mario's jump covers roughly 48 pixels horizontally and he lands approximately at the same vertical level
    landing_zone_x_start = enemy_x + 30  # Right edge of the enemy
    landing_zone_x_end = landing_zone_x_start + 40  # Width of Mario's jump
    landing_zone_y_start = mario_y - 16  # A bit above Mario's current level
    landing_zone_y_end = mario_y + 16  # A bit below Mario's current level

    # Check for other enemies in the landing zone
    for other_enemy in all_enemies:
        print("THis is the value of other enemy", other_enemy)
        other_enemy_x = other_enemy[0][0]
        print("This is the value of other_enemy_x", other_enemy_x)
        other_enemy_y = other_enemy[0][1]
        if landing_zone_x_start <= other_enemy_x <= landing_zone_x_end and landing_zone_y_start <= other_enemy_y <= landing_zone_y_end:
            return False  # There's another enemy in the landing zone

    return True

def can_jump_over(screen, mario_location, enemy_location, threshold_distance=32):
    """
    Determine if Mario can jump over an enemy based on obstacles above him.

    Parameters:
    - screen: The current game screen/frame.
    - mario_location: Tuple containing Mario's current (x, y) coordinates.
    - enemy_location: Tuple containing the enemy's (x, y) coordinates.
    - threshold_distance: The vertical distance to check above Mario for obstacles. Default is 32 pixels.

    Returns:
    - True if Mario can jump over the enemy. False otherwise.
    """

    # Get the x and y coordinates of Mario
    mario_x, mario_y = mario_location

    # Get the x coordinate of the enemy
    enemy_x, _ = enemy_location

    # Check if the enemy is within a reasonable horizontal distance from Mario
    if abs(mario_x - enemy_x) > threshold_distance:
        return False  # Enemy is too far away to consider jumping over

    # Define the region of interest above Mario
    roi = screen[mario_y - threshold_distance:mario_y, mario_x - 16:mario_x + 16]

    # Use the object detection mechanism to detect blocks or platforms in the region of interest
    detected_objects = locate_objects(roi, "small")  # Assuming Mario is small; adjust as needed

    # Check for blocks or platforms in the detected objects
    for category, objects in detected_objects.items():
        if category == "block" and objects:
            return False  # There's a block above Mario, so he can't jump over

    return True  # No obstacles detected above Mario

def is_enemy_behind(screen, mario_location, object_locations, threshold_distance=32):
    """
    Determine if there's an enemy behind Mario within a certain distance.

    Parameters:
    - screen: The current game screen/frame.
    - mario_location: Tuple containing Mario's current (x, y) coordinates.
    - object_locations: Dictionary containing locations of various objects on the screen.
    - threshold_distance: The horizontal distance to check behind Mario for enemies. Default is 32 pixels.

    Returns:
    - True if there's an enemy behind Mario within the threshold distance. False otherwise.
    """

    # Get the x and y coordinates of Mario
    mario_x, mario_y = mario_location

    # Define the region of interest behind Mario
    roi = screen[mario_y - 16:mario_y + 32, mario_x - threshold_distance:mario_x]

    # Use the object detection mechanism to detect enemies in the region of interest
    detected_objects = locate_objects(roi, "small")  # Assuming Mario is small; adjust as needed

    # Check for enemies in the detected objects
    if "enemy" in detected_objects and detected_objects["enemy"]:
        return True  # There's an enemy behind Mario

    return False  # No enemy detected behind Mario
################################################################################
#Pipe helper functions
def is_safe_to_jump_over_pipe(pipe, enemies, safe_distance=40):
    """
    Determines if it's safe to jump over a pipe by checking for enemies.

    Parameters:
    - pipe: The pipe's location and dimensions.
    - enemies: List of enemies' locations and dimensions.
    - safe_distance: The minimum distance required after the pipe to consider it safe.

    Returns:
    - True if it's safe to jump over the pipe, False otherwise.
    """
    pipe_x = pipe[0][0]
    pipe_y = pipe[0][1]
    pipe_width = pipe[1][0]

    landing_zone_x = pipe_x + pipe_width + safe_distance

    for enemy in enemies:
        enemy_x, enemy_y = enemy[0]
        if pipe_x + pipe_width < enemy_x < landing_zone_x:
            return False

    return True

def is_on_top_of_pipe(mario_location, pipe):
    """
    Determines if Mario is on top of the given pipe.

    Parameters:
    - mario_location: Tuple containing Mario's (x, y) coordinates.
    - pipe: Tuple containing the pipe's location and dimensions.

    Returns:
    - True if Mario is on top of the pipe. False otherwise.
    """

    mario_x = mario_location[0][0]
    mario_y = mario_location[0][1]
    print("The pipe charectersitxs we are checkig we are ontop of are", pipe)
    pipe_x = pipe[0][0]
    pipe_y = pipe[0][1]
    pipe_width = pipe[1][0]


    # Check if Mario's x-coordinate is within the horizontal bounds of the pipe
    if mario_x >= pipe_x and mario_x <= pipe_x + pipe_width:
        # Check if Mario's y-coordinate is just above the top of the pipe (with a small threshold for error)
        if abs(mario_y - pipe_y) < 5:  # 5 is an arbitrary threshold; adjust as needed
            return True

    return False

def action_from_pipe_top(pipe, enemy_locations):
    """
    Determines the best action for Mario when he's on top of a pipe.

    Parameters:
    - pipe: Tuple containing the pipe's location and dimensions.
    - enemy_locations: List of tuples containing the locations and dimensions of all detected enemies.

    Returns:
    - An action for Mario to take.
    """

    pipe_x, pipe_y = pipe[0]
    pipe_width, pipe_height = pipe[1]

    # Define a region just below the pipe where we'll check for enemies
    danger_zone = (pipe_x, pipe_y + pipe_height, pipe_width, 20)  # 20 is an arbitrary height; adjust as needed

    for enemy in enemy_locations:
        enemy_x, enemy_y = enemy[0]
        enemy_width, enemy_height = enemy[1]

        # Check if any part of the enemy overlaps with the danger zone
        if (enemy_x + enemy_width > danger_zone[0] and enemy_x < danger_zone[0] + danger_zone[2] and
            enemy_y + enemy_height > danger_zone[1] and enemy_y < danger_zone[1] + danger_zone[3]):
            # If an enemy is detected in the danger zone, jump and move right
            return 4  # ['right', 'A', 'B']

    # If no enemies are detected in the danger zone, simply move right
    return 1  # ['right']

################################################################################
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

obs = None
done = True
env.reset()
initial_lives = 2
for step in range(100000):
    if obs is not None:
        action = make_action(obs, info, step, env, action)
    else:
        action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    #if done:
    #    break    #SET TO env.reset() IF WE WANT SIMULATION TO KEEP RUNNING
    if info['life'] < initial_lives:
        break  # End the simulation if Mario loses a life
env.close()
