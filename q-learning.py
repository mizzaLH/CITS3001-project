from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string
import os
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
#######################################
#Enemy Helper Functions
def is_high_threat(mario_locations, enemy_locations, threat_distance=30):
    """
    Determines if an enemy is a high threat based on its proximity to Mario.

    Parameters:
    - mario_location: Tuple containing Mario's (x, y) coordinates.
    - enemy_location: Tuple containing the enemy's (x, y) coordinates.
    - threat_distance: The horizontal distance within which an enemy is considered a high threat.

    Returns:
    - True if the enemy is a high threat. False otherwise.
    """
    print(f"mario_locations: {mario_locations}, enemy_locations: {enemy_locations}")

    if mario_locations and enemy_locations:
        mario_location = mario_locations[0]
        enemy_location = enemy_locations[0]

        # 从元组中提取x和y坐标
        mario_x, mario_y = mario_location[0]
        enemy_x, enemy_y = enemy_location[0]

    # Calculate horizontal and vertical distances between Mario and the enemy
    horizontal_distance = abs(mario_x - enemy_x)
    vertical_distance = abs(mario_y - enemy_y)

    # Check if the enemy is within the threat distance horizontally
    # and is approximately at the same vertical level as Mario.
    if horizontal_distance < threat_distance and vertical_distance < 20:
        print("have found an enemy")
        return True

    return False

def is_enemy_behind_enemy(enemy_locations):
    """
    Check if there is another enemy within two blocks behind the first enemy.
    
    Parameters:
    mario_location (tuple): The location of Mario as a tuple (x, y).
    enemy_locations (list): A list of tuples each containing enemy location and size information.
    
    Returns:
    bool: True if there is another enemy within two blocks behind the first enemy, False otherwise.
    """

    first_enemy_location = enemy_locations[0] if enemy_locations else None

    if first_enemy_location is None:
        # 如果没有敌人，返回 False
        return False

    first_enemy_x, first_enemy_y = first_enemy_location[0]

    # 定义检查区域的范围
    check_range_start = first_enemy_x + 16  # 一个 block 后的位置
    check_range_end = first_enemy_x + 32   # 两个 block 后的位置（两个 block 之外）

    # 遍历所有的敌人，检查是否有敌人在指定的范围内
    for enemy_location in enemy_locations[1:]:  # 跳过第一个敌人
        enemy_x, enemy_y = enemy_location[0]
        if check_range_start <= enemy_x <= check_range_end:
            # 如果找到一个敌人在指定的范围内，返回 True
            print("it's an enemy behind the first enemy")
            return True

    # 如果遍历完所有的敌人都没有找到在指定范围内的敌人，返回 False
    return False

def is_two_enemies_behind_the_second_enemy(enemy_locations):
    """
    Check if there are at least two enemies within six blocks behind the second enemy.
    
    Parameters:
    enemy_locations (list): A list of tuples each containing enemy location and size information.
    
    Returns:
    bool: True if there are at least two enemies within six blocks behind the second enemy, False otherwise.
    """

    # 确保有至少三个敌人（包括第二个敌人）在列表中
    if len(enemy_locations) < 3:
        return False

    # 获取第二个敌人的位置
    second_enemy_x, second_enemy_y = enemy_locations[1][0]

    # 定义检查区域的范围
    check_range_start = second_enemy_x + 16  # 一个 block 后的位置
    check_range_end = second_enemy_x + 112  # 七个 block 后的位置（六个 block 之外）

    # 初始化一个计数器来跟踪在指定范围内找到的敌人数量
    enemies_count = 0

    # 遍历所有的敌人，检查是否有敌人在指定的范围内
    for enemy_location in enemy_locations[2:]:  # 跳过前两个敌人
        enemy_x, enemy_y = enemy_location[0]
        if check_range_start <= enemy_x <= check_range_end:
            # 如果找到一个敌人在指定的范围内，增加计数器
            enemies_count += 1

            # 如果找到至少两个敌人，返回 True
            if enemies_count >= 2:
                return True

    # 如果遍历完所有的敌人都没有找到至少两个敌人在指定范围内，返回 False
    return False

def is_pipe(mario_locations, pipe_locations):
    """
    Check if there is a pipe in front of Mario.
    
    Parameters:
    mario_locations (tuple): The location of Mario as a tuple (x, y).
    pipe_locations (list): A list of tuples each containing pipe location and size information.
    
    Returns:
    bool: True if there is a pipe in front of Mario, False otherwise.
    """

    # 获取 Mario 的 x 和 y 坐标
    mario_x, mario_y = mario_locations

    for pipe in pipe_locations:
        pipe_x, pipe_y = pipe[0]
        pipe_width, pipe_height = pipe[1]
        
        # 检查管道是否在 Mario 的前方
        # 我们可以通过比较 x 坐标来实现这一点
        # 我们也需要确保 Mario 和管道的 y 坐标相近，以便确保它们在同一水平线上
        if pipe_x > mario_x and abs(pipe_y - mario_y) < pipe_height and pipe_x - mario_x < 24:
            return True  # 如果找到了管道，返回 True
    
    return False  # 如果遍历完所有的管道都没有找到符合条件的管道，返回 False



def is_on_top_of_pipe(mario_locations, pipe):
    """
    Determines if Mario is on top of the given pipe.

    Parameters:
    - mario_location: Tuple containing Mario's (x, y) coordinates.
    - pipe: Tuple containing the pipe's location and dimensions.

    Returns:
    - True if Mario is on top of the pipe. False otherwise.
    """
    if mario_locations:
        mario_location = mario_locations[0]
        # 从元组中提取x和y坐标
        mario_x = mario_location[0][0]
        mario_y = mario_location[0][1]
    print("The pipe charectersitxs we are checkig we are on top of are", pipe)
    pipe_x = pipe[0][0]
    pipe_y = pipe[0][1]
    pipe_width = pipe[1][0]


    # Check if Mario's x-coordinate is within the horizontal bounds of the pipe
    if mario_x >= pipe_x and mario_x <= pipe_x + pipe_width:
        # Check if Mario's y-coordinate is just above the top of the pipe (with a small threshold for error)
        if abs(mario_y - pipe_y) < 5:  # 5 is an arbitrary threshold; adjust as needed
            return True

    return False

def is_gap(mario_locations, block_locations):
    """
    Check if there is a gap in the ground in front of Mario.
    
    Parameters:
    mario_locations (tuple): The location of Mario as a tuple (x, y).
    block_locations (list): A list of tuples each containing block location, size, and name information.
    
    Returns:
    bool: True if there is a gap in front of Mario, False otherwise.
    """
    
    # 获取马里奥的 x 坐标
    if mario_locations:
        mario_location = mario_locations[0]
        # 从元组中提取x和y坐标
        mario_x, mario_y = mario_location[0]

    # 初始化前一个 block2 的 x 坐标为 None，以便我们可以在循环中更新它
    prev_block2_x = None

    # 遍历所有的 block，查找名为 'block2' 的 block
    for block in block_locations:
        block_x = block[0][0]
        block_y = block[0][1]
        block_name = block[2]

        if block_name == 'block2' and block_y == 224:
            # 如果这是第一个找到的 'block2' block，更新 prev_block2_x 并继续
            if prev_block2_x is None:
                prev_block2_x = block_x
                continue

            # 计算当前 'block2' block 和前一个 'block2' block 之间的 x 坐标差
            x_gap = block_x - prev_block2_x

            # 检查 x 坐标差是否大于 16，并且这个 gap 是否在马里奥的前方
            if x_gap > 16 and mario_x + 24 >= prev_block2_x + 16 and mario_x <= prev_block2_x:
                print("在马里奥前方发现了一个 gap")
                return True  # 在马里奥前方发现了一个 gap

            # 更新 prev_block2_x 为当前 'block2' block 的 x 坐标，以便下一次迭代
            prev_block2_x = block_x

    # 没有在马里奥前方发现 gap，返回 False
    return False

def is_block4(mario_locations, block_locations):
    """
    检查 Mario 前方 0.5 格内是否有 block4 障碍物。

    参数:
    mario_locations (tuple): Mario 的位置信息。
    block_locations (list): 所有 block 的位置信息列表。

    返回:
    bool: 如果前方 0.5 格内有 block4，则返回 True，否则返回 False。
    """

    mario_x, mario_y = mario_locations[0][0]

    # 计算检查区域
    check_range_start = mario_x
    check_range_end = mario_x + 15  # 0.5格宽度为8像素

    for block in block_locations:
        block_x = block[0][0]
        block_y = block[0][1]
        block_name = block[2]

        if block_name == 'block4' and check_range_start <= block_x <= check_range_end:
            return True  # 在检查区域内找到 block4

    return False  # 没有在检查区域内找到 block4


def is_on_the_top_of_block4(mario_locations, block_locations):
    """
    检查 Mario 是否站在最顶端的 block4 砖块上。

    参数:
    mario_locations (tuple): Mario 的位置信息。
    block_locations (list): 所有 block 的位置信息列表。

    返回:
    bool: 如果 Mario 站在最顶端的 block4 砖块上，则返回 True，否则返回 False。
    """

    if not is_block4(mario_locations, block_locations):
        # 如果前方没有 block4，则检查 Mario 是否站在 block4 上
        mario_x, mario_y = mario_locations[0][0]

        for block in block_locations:
            block_x, block_y = block[0]
            block_name = block[2]

            if block_name == 'block4' and block_y >= mario_y + 16:  # 假设 Mario 的高度为16像素
                return True  # Mario 站在 block4 上

    return False  # Mario 没有站在 block4 上或前方有 block4



################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION
def get_state_index(object_locations):
    mario_locations = object_locations["mario"]  
    enemy_locations = object_locations["enemy"]
    block_locations = object_locations["block"]
    pipe_locations = object_locations["pipe"]
    
    # mario_x, mario_y = mario_locations
    # enemy_x, enemy_y = enemy_locations
    # we can define state nothing for number 20
    # for state(1,2,3) for dealing with the enemies | State 1 only has one enemy in 6 blocks (assume)
    # see if there is a enemy in front of Mario
    if mario_locations:
        if enemy_locations:
            if is_high_threat(mario_locations, enemy_locations): 
                # see if the enemy is the only one, so we should detect if there any other enemy after the first one in 6 blocks
                if is_enemy_behind_enemy(enemy_locations):
                    # if not then we return the state 1
                    # if yes then we will find out if it's the state of two enemies
                        # if yes then we return the state 2
                        # if not then we will find out if it's the state of four enemies
                            #if yes then we return the state 3
                            #if not, it's impossible not usually, then we return nothing and print out here's a bug
                    if is_two_enemies_behind_the_second_enemy(enemy_locations):
                        print("state 3")
                        return 3
                    print("state 2")
                    return 2
                print("state 1")        
                return 1
        
        # see if the Mario is standing on the top pipe
        if pipe_locations:
            for pipe in pipe_locations:
                pipe_x = pipe[0][0]
            if is_on_top_of_pipe(mario_locations,pipe):
                return 5
            if is_pipe:
                return 4
            # if yes then we return state 5
        # see if there is a pipe in front of Mario, and the distance to pipe should be closed, maybe 1 or 1.5 blocks:
            # if yes then we return state 4

        # see if there is a gap in front of Mario, and the distance maybe 1 or 1.5 block
        if is_gap(mario_locations, block_locations):
            return 6
            # if yes then we return state 6

        # see if there is a block 4 barrier in front of Mario:
        if is_block4(mario_locations, block_locations):
            return 7
            # if yes then we return state 7
        
        # see if Mario stand on the edge block of block 4
        if is_on_the_top_of_block4(mario_locations, block_locations):
            return 8
            # if yes then we return state 8

        
    return 0
    # enemy_in_front = any(
    #     (enemy[0][0] > mario[0][0] and abs(enemy[0][1] - mario[0][1]) < 20)
    #     for enemy in object_locations["enemy"]
    #     for mario in object_locations["mario"]
    # )
    # pipe_in_front = any(
    #     (pipe[0][0] > mario[0][0] and abs(pipe[0][1] - mario[0][1]) < 20)
    #     for pipe in object_locations["pipe"]
    #     for mario in object_locations["mario"]
    # )
    # # 根据二进制编码计算状态索引
    # return int(f"{int(enemy_in_front)}{int(pipe_in_front)}", 2)


def make_action(screen, info, step, env, prev_action):
    global current_state_index, current_action
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
    current_state_index = get_state_index(object_locations)  # 更新当前状态索引
    print("current_state_index")
    print(current_state_index)
    if np.random.uniform(0, 1) < exploration_rate:
        current_action = env.action_space.sample()  # 随机选择动作
    else:
        print("choose action by")
        print(current_action)
        current_action = np.argmax(q_table[current_state_index, :])  # 选择Q值最高的动作
    return current_action

################################################################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

state_space_size = 10  # 根据你的状态空间大小填写
action_space_size = 12  # 从你的动作集合中获得
# Check if Q-table file exists
if os.path.exists('q_table.npy'):
    # Load Q-table from file
    q_table = np.load('q_table.npy')
else:
    # Initialize a new Q-table
    q_table = np.zeros((state_space_size, action_space_size))

learning_rate = 0.01
discount_factor = 0.99
exploration_rate = 0.5
max_exploration_rate = 1
min_exploration_rate = 0.1
exploration_decay_rate = 0.01

current_state_index = None
current_action = None

obs = None
done = True
env.reset()
for step in range(100000):
    if obs is not None:
        action = make_action(obs, info, step, env, action)
    else:
        action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("reward")
    print(reward)
     # 新代码: 获取下一个状态的索引
    object_locations = locate_objects(obs, info["status"])  # 获取物体的新位置
    next_state_index = get_state_index(object_locations)  # 根据新的物体位置计算下一个状态的索引

    # 新代码: 如果这不是第一步，更新Q-table
    # redefine reward + = new reward 
    # and punishment


    if current_state_index is not None and current_action is not None:
        q_table[current_state_index, current_action] = q_table[current_state_index, current_action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_factor * np.max(q_table[next_state_index, :]))
    print(q_table)
    np.save('q_table.npy', q_table)

    done = terminated or truncated
    if done:
        env.reset()
env.close()
