import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/sort/trainlength40/s0/sort_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 45, 46}:
            return k_position == 3
        elif q_position in {40, 1, 42}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 5}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {38, 6, 39}:
            return k_position == 11
        elif q_position in {11, 7}:
            return k_position == 8
        elif q_position in {8, 33}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {12, 13}:
            return k_position == 7
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16, 37}:
            return k_position == 21
        elif q_position in {17}:
            return k_position == 22
        elif q_position in {49, 18}:
            return k_position == 24
        elif q_position in {19}:
            return k_position == 17
        elif q_position in {20, 47}:
            return k_position == 4
        elif q_position in {21, 23}:
            return k_position == 27
        elif q_position in {35, 22}:
            return k_position == 26
        elif q_position in {24, 30}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 29
        elif q_position in {26}:
            return k_position == 30
        elif q_position in {41, 27}:
            return k_position == 32
        elif q_position in {28}:
            return k_position == 16
        elif q_position in {29}:
            return k_position == 13
        elif q_position in {36, 31}:
            return k_position == 25
        elif q_position in {32}:
            return k_position == 34
        elif q_position in {34}:
            return k_position == 35
        elif q_position in {43}:
            return k_position == 12
        elif q_position in {44}:
            return k_position == 45
        elif q_position in {48}:
            return k_position == 15

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 16
        elif q_position in {1, 2}:
            return k_position == 4
        elif q_position in {16, 3, 6}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {17, 12, 5}:
            return k_position == 11
        elif q_position in {20, 7}:
            return k_position == 13
        elif q_position in {8, 27, 21}:
            return k_position == 12
        elif q_position in {32, 36, 9, 11, 15}:
            return k_position == 3
        elif q_position in {10, 37, 14, 22}:
            return k_position == 8
        elif q_position in {13}:
            return k_position == 18
        elif q_position in {33, 18, 24, 30, 31}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 23
        elif q_position in {35, 47, 23}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 30
        elif q_position in {26}:
            return k_position == 31
        elif q_position in {28}:
            return k_position == 37
        elif q_position in {29}:
            return k_position == 34
        elif q_position in {34}:
            return k_position == 29
        elif q_position in {38}:
            return k_position == 1
        elif q_position in {39}:
            return k_position == 6
        elif q_position in {40}:
            return k_position == 38
        elif q_position in {41}:
            return k_position == 35
        elif q_position in {42}:
            return k_position == 43
        elif q_position in {43}:
            return k_position == 42
        elif q_position in {44, 46}:
            return k_position == 15
        elif q_position in {45}:
            return k_position == 39
        elif q_position in {48}:
            return k_position == 45
        elif q_position in {49}:
            return k_position == 10

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 4, 5, 39, 24, 27}:
            return token == "2"
        elif position in {32, 1, 20, 29}:
            return token == "0"
        elif position in {2, 28, 31}:
            return token == "1"
        elif position in {3, 7, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {34, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23}:
            return token == "4"
        elif position in {36, 37, 11, 22, 25, 26}:
            return token == "3"
        elif position in {35, 30}:
            return token == "<s>"
        elif position in {33, 38}:
            return token == "</s>"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 22, 31}:
            return token == "3"
        elif position in {32, 1, 33, 39, 16, 17, 23, 26}:
            return token == "0"
        elif position in {2, 3, 4, 5, 6, 7, 8, 38, 18, 24, 25}:
            return token == "2"
        elif position in {35, 37, 9, 10, 11, 12, 13, 14, 20, 21, 27, 28, 30}:
            return token == "4"
        elif position in {19, 15}:
            return token == "1"
        elif position in {34, 36, 29}:
            return token == "</s>"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 8, 5}:
            return k_position == 4
        elif q_position in {1, 27}:
            return k_position == 9
        elif q_position in {2, 4, 39, 13, 15, 25}:
            return k_position == 3
        elif q_position in {34, 3}:
            return k_position == 2
        elif q_position in {17, 6}:
            return k_position == 8
        elif q_position in {10, 7}:
            return k_position == 11
        elif q_position in {9, 22}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {43, 14}:
            return k_position == 17
        elif q_position in {16, 21}:
            return k_position == 10
        elif q_position in {18}:
            return k_position == 28
        elif q_position in {24, 19}:
            return k_position == 29
        elif q_position in {20, 28}:
            return k_position == 26
        elif q_position in {23}:
            return k_position == 30
        elif q_position in {26}:
            return k_position == 7
        elif q_position in {29}:
            return k_position == 36
        elif q_position in {32, 30}:
            return k_position == 35
        elif q_position in {31}:
            return k_position == 5
        elif q_position in {33, 35}:
            return k_position == 6
        elif q_position in {36}:
            return k_position == 0
        elif q_position in {37}:
            return k_position == 23
        elif q_position in {38}:
            return k_position == 18
        elif q_position in {40, 47}:
            return k_position == 42
        elif q_position in {41}:
            return k_position == 45
        elif q_position in {42}:
            return k_position == 33
        elif q_position in {44}:
            return k_position == 38
        elif q_position in {45}:
            return k_position == 47
        elif q_position in {46}:
            return k_position == 34
        elif q_position in {48}:
            return k_position == 40
        elif q_position in {49}:
            return k_position == 27

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {23, 7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9, 45}:
            return k_position == 11
        elif q_position in {41, 10, 11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {49, 13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17, 43}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {19, 39}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {24}:
            return k_position == 31
        elif q_position in {25}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 27
        elif q_position in {27}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 24
        elif q_position in {29}:
            return k_position == 30
        elif q_position in {30}:
            return k_position == 34
        elif q_position in {32, 48, 46, 31}:
            return k_position == 33
        elif q_position in {33}:
            return k_position == 8
        elif q_position in {34}:
            return k_position == 35
        elif q_position in {35, 36}:
            return k_position == 37
        elif q_position in {37}:
            return k_position == 38
        elif q_position in {38}:
            return k_position == 39
        elif q_position in {40}:
            return k_position == 41
        elif q_position in {42, 47}:
            return k_position == 47
        elif q_position in {44}:
            return k_position == 46

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {
            0,
            4,
            6,
            7,
            11,
            12,
            13,
            15,
            16,
            17,
            23,
            25,
            26,
            27,
            29,
            30,
            31,
            32,
            34,
        }:
            return token == "4"
        elif position in {1, 19}:
            return token == "0"
        elif position in {9, 2, 3, 39}:
            return token == "1"
        elif position in {5}:
            return token == "3"
        elif position in {35, 36, 37, 8, 10, 14, 18, 20, 21, 22, 24, 28}:
            return token == "2"
        elif position in {33}:
            return token == "</s>"
        elif position in {38}:
            return token == "<s>"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1", "</s>", "<s>", "3"}:
            return k_token == "3"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "4"

    attn_0_7_pattern = select_closest(tokens, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 14}:
            return k_position == 17
        elif q_position in {1, 12}:
            return k_position == 15
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {41, 5, 6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 19
        elif q_position in {17, 18}:
            return k_position == 22
        elif q_position in {19}:
            return k_position == 23
        elif q_position in {20, 46}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 25
        elif q_position in {22, 39}:
            return k_position == 26
        elif q_position in {23}:
            return k_position == 27
        elif q_position in {24, 25}:
            return k_position == 29
        elif q_position in {40, 26, 44}:
            return k_position == 30
        elif q_position in {27}:
            return k_position == 32
        elif q_position in {42, 28, 30}:
            return k_position == 34
        elif q_position in {29}:
            return k_position == 33
        elif q_position in {31}:
            return k_position == 36
        elif q_position in {32, 33}:
            return k_position == 37
        elif q_position in {34}:
            return k_position == 39
        elif q_position in {35}:
            return k_position == 42
        elif q_position in {36}:
            return k_position == 44
        elif q_position in {37}:
            return k_position == 46
        elif q_position in {38}:
            return k_position == 40
        elif q_position in {49, 43, 45, 47}:
            return k_position == 0
        elif q_position in {48}:
            return k_position == 35

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 34, 28, 31}:
            return token == "1"
        elif position in {1, 2, 3, 4, 7, 40, 9, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {5, 39, 10, 13, 14}:
            return token == "<s>"
        elif position in {6}:
            return token == "<pad>"
        elif position in {8, 11, 12, 15, 16, 17, 18, 19}:
            return token == "</s>"
        elif position in {
            32,
            33,
            35,
            36,
            37,
            38,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            29,
            30,
        }:
            return token == "0"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 10, 11}:
            return k_position == 17
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2, 3}:
            return k_position == 6
        elif q_position in {42, 4, 5, 44}:
            return k_position == 9
        elif q_position in {6, 39}:
            return k_position == 11
        elif q_position in {7}:
            return k_position == 13
        elif q_position in {8, 43}:
            return k_position == 14
        elif q_position in {9}:
            return k_position == 15
        elif q_position in {40, 12, 47}:
            return k_position == 20
        elif q_position in {13}:
            return k_position == 21
        elif q_position in {14, 15}:
            return k_position == 23
        elif q_position in {16}:
            return k_position == 26
        elif q_position in {17}:
            return k_position == 25
        elif q_position in {18}:
            return k_position == 28
        elif q_position in {19}:
            return k_position == 27
        elif q_position in {20}:
            return k_position == 29
        elif q_position in {21, 22}:
            return k_position == 31
        elif q_position in {23}:
            return k_position == 33
        elif q_position in {24, 25}:
            return k_position == 35
        elif q_position in {26}:
            return k_position == 36
        elif q_position in {27}:
            return k_position == 38
        elif q_position in {28}:
            return k_position == 37
        elif q_position in {29}:
            return k_position == 39
        elif q_position in {38, 30}:
            return k_position == 45
        elif q_position in {33, 35, 36, 31}:
            return k_position == 47
        elif q_position in {32}:
            return k_position == 44
        elif q_position in {34}:
            return k_position == 43
        elif q_position in {37}:
            return k_position == 49
        elif q_position in {48, 41}:
            return k_position == 18
        elif q_position in {45}:
            return k_position == 12
        elif q_position in {46}:
            return k_position == 19
        elif q_position in {49}:
            return k_position == 1

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {
            0,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            41,
            44,
            45,
            46,
        }:
            return token == ""
        elif position in {1}:
            return token == "</s>"
        elif position in {2, 3, 4, 5, 6, 7, 8, 40, 42, 43, 47, 48, 49}:
            return token == "0"
        elif position in {9}:
            return token == "<s>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0, 4, 5, 13, 45}:
            return k_position == 18
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2, 47, 7}:
            return k_position == 11
        elif q_position in {3, 12}:
            return k_position == 16
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {39, 15}:
            return k_position == 20
        elif q_position in {16, 17, 42}:
            return k_position == 21
        elif q_position in {18}:
            return k_position == 24
        elif q_position in {41, 19}:
            return k_position == 25
        elif q_position in {20, 44}:
            return k_position == 26
        elif q_position in {21}:
            return k_position == 27
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 30
        elif q_position in {24}:
            return k_position == 31
        elif q_position in {25, 28, 46}:
            return k_position == 32
        elif q_position in {26}:
            return k_position == 33
        elif q_position in {27}:
            return k_position == 35
        elif q_position in {29}:
            return k_position == 36
        elif q_position in {30}:
            return k_position == 37
        elif q_position in {31}:
            return k_position == 38
        elif q_position in {32, 33, 38}:
            return k_position == 40
        elif q_position in {34}:
            return k_position == 45
        elif q_position in {35, 37}:
            return k_position == 44
        elif q_position in {36}:
            return k_position == 48
        elif q_position in {40, 49, 43}:
            return k_position == 1
        elif q_position in {48}:
            return k_position == 10

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1}:
            return token == "</s>"
        elif position in {2, 3, 4, 5, 6, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == "0"
        elif position in {7}:
            return token == "<s>"
        elif position in {
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
        }:
            return token == ""
        elif position in {39}:
            return token == "4"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 1, 2, 3, 6, 7, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49}:
            return token == ""
        elif position in {48, 4}:
            return token == "<pad>"
        elif position in {16, 11, 20, 5}:
            return token == "<s>"
        elif position in {8, 9, 10, 12, 13, 14, 17}:
            return token == "</s>"
        elif position in {32, 33, 35, 37, 15, 19, 21, 22, 23, 25, 26, 27, 29, 30}:
            return token == "1"
        elif position in {34, 36, 38, 18, 24, 28, 31}:
            return token == "0"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {
            0,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            40,
            42,
            43,
            44,
            45,
            46,
        }:
            return token == ""
        elif position in {1}:
            return token == "</s>"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6, 7, 8, 41, 47, 48, 49}:
            return token == "1"
        elif position in {39, 9, 10, 11, 12, 13, 14}:
            return token == "0"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {4}:
            return 39
        return 13

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_4_output, attn_0_2_output):
        key = (attn_0_4_output, attn_0_2_output)
        if key in {("0", "0"), ("</s>", "0")}:
            return 38
        return 30

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_2_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_1_output, attn_0_5_output):
        key = (attn_0_1_output, attn_0_5_output)
        return 23

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_5_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_0_output):
        key = attn_0_0_output
        return 47

    mlp_0_3_outputs = [mlp_0_3(k0) for k0 in attn_0_0_outputs]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 47

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_4_output, num_attn_0_2_output):
        key = (num_attn_0_4_output, num_attn_0_2_output)
        return 7

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_6_output, num_attn_0_0_output):
        key = (num_attn_0_6_output, num_attn_0_0_output)
        return 28

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_4_output, num_attn_0_2_output):
        key = (num_attn_0_4_output, num_attn_0_2_output)
        return 41

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, mlp_0_0_output):
        if position in {
            0,
            32,
            40,
            42,
            43,
            12,
            13,
            44,
            45,
            16,
            47,
            18,
            48,
            49,
            25,
            26,
            28,
        }:
            return mlp_0_0_output == 5
        elif position in {1, 34, 35, 38, 39, 10}:
            return mlp_0_0_output == 3
        elif position in {33, 2, 3, 9, 41, 14, 46, 17, 20, 21, 23, 29}:
            return mlp_0_0_output == 13
        elif position in {4, 5, 6, 7, 11, 27, 30, 31}:
            return mlp_0_0_output == 4
        elif position in {8}:
            return mlp_0_0_output == 2
        elif position in {15}:
            return mlp_0_0_output == 39
        elif position in {19}:
            return mlp_0_0_output == 18
        elif position in {22}:
            return mlp_0_0_output == 25
        elif position in {24}:
            return mlp_0_0_output == 44
        elif position in {36}:
            return mlp_0_0_output == 6
        elif position in {37}:
            return mlp_0_0_output == 1

    attn_1_0_pattern = select_closest(mlp_0_0_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_7_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0, 32, 34, 35, 6, 7, 9, 11, 16, 22, 23, 25, 30, 31}:
            return token == "4"
        elif position in {1, 42, 44, 45, 18}:
            return token == "0"
        elif position in {2, 19}:
            return token == "1"
        elif position in {3, 39, 43, 49, 29}:
            return token == ""
        elif position in {33, 4, 5, 36, 37, 38, 40, 10, 14, 47, 17, 21}:
            return token == "2"
        elif position in {8, 12, 13, 15, 20, 26, 27}:
            return token == "3"
        elif position in {24, 48, 46}:
            return token == "</s>"
        elif position in {41, 28}:
            return token == "<s>"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_0_output, mlp_0_3_output):
        if mlp_0_0_output in {0, 35, 42, 44, 16, 17, 48, 49, 20, 21, 24, 27}:
            return mlp_0_3_output == 5
        elif mlp_0_0_output in {1, 9, 15}:
            return mlp_0_3_output == 15
        elif mlp_0_0_output in {2, 10}:
            return mlp_0_3_output == 17
        elif mlp_0_0_output in {33, 3, 28, 23}:
            return mlp_0_3_output == 6
        elif mlp_0_0_output in {4}:
            return mlp_0_3_output == 24
        elif mlp_0_0_output in {36, 5, 6, 7, 37, 40, 30}:
            return mlp_0_3_output == 0
        elif mlp_0_0_output in {8}:
            return mlp_0_3_output == 16
        elif mlp_0_0_output in {11, 46, 47}:
            return mlp_0_3_output == 3
        elif mlp_0_0_output in {12, 39}:
            return mlp_0_3_output == 13
        elif mlp_0_0_output in {13}:
            return mlp_0_3_output == 47
        elif mlp_0_0_output in {14}:
            return mlp_0_3_output == 39
        elif mlp_0_0_output in {25, 18}:
            return mlp_0_3_output == 2
        elif mlp_0_0_output in {19}:
            return mlp_0_3_output == 45
        elif mlp_0_0_output in {22}:
            return mlp_0_3_output == 7
        elif mlp_0_0_output in {26, 31}:
            return mlp_0_3_output == 14
        elif mlp_0_0_output in {29}:
            return mlp_0_3_output == 11
        elif mlp_0_0_output in {32, 41}:
            return mlp_0_3_output == 25
        elif mlp_0_0_output in {34, 43}:
            return mlp_0_3_output == 4
        elif mlp_0_0_output in {38}:
            return mlp_0_3_output == 28
        elif mlp_0_0_output in {45}:
            return mlp_0_3_output == 29

    attn_1_2_pattern = select_closest(mlp_0_3_outputs, mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_7_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_4_output, mlp_0_0_output):
        if attn_0_4_output in {"0", "3"}:
            return mlp_0_0_output == 1
        elif attn_0_4_output in {"1"}:
            return mlp_0_0_output == 33
        elif attn_0_4_output in {"2"}:
            return mlp_0_0_output == 0
        elif attn_0_4_output in {"4"}:
            return mlp_0_0_output == 13
        elif attn_0_4_output in {"</s>"}:
            return mlp_0_0_output == 12
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_0_output == 2

    attn_1_3_pattern = select_closest(mlp_0_0_outputs, attn_0_4_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_5_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_7_output, token):
        if attn_0_7_output in {"0"}:
            return token == "3"
        elif attn_0_7_output in {"1", "<s>"}:
            return token == ""
        elif attn_0_7_output in {"2"}:
            return token == "1"
        elif attn_0_7_output in {"3"}:
            return token == "4"
        elif attn_0_7_output in {"4"}:
            return token == "2"
        elif attn_0_7_output in {"</s>"}:
            return token == "<s>"

    attn_1_4_pattern = select_closest(tokens, attn_0_7_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_0_output, num_mlp_0_1_output):
        if attn_0_0_output in {"0"}:
            return num_mlp_0_1_output == 4
        elif attn_0_0_output in {"1"}:
            return num_mlp_0_1_output == 0
        elif attn_0_0_output in {"</s>", "2"}:
            return num_mlp_0_1_output == 33
        elif attn_0_0_output in {"3"}:
            return num_mlp_0_1_output == 25
        elif attn_0_0_output in {"4"}:
            return num_mlp_0_1_output == 7
        elif attn_0_0_output in {"<s>"}:
            return num_mlp_0_1_output == 6

    attn_1_5_pattern = select_closest(
        num_mlp_0_1_outputs, attn_0_0_outputs, predicate_1_5
    )
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_4_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 4
        elif q_position in {1, 9}:
            return k_position == 1
        elif q_position in {27, 3, 37}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {26, 5}:
            return k_position == 23
        elif q_position in {6, 22}:
            return k_position == 14
        elif q_position in {39, 7}:
            return k_position == 3
        elif q_position in {36, 8, 41, 40, 46, 15, 48}:
            return k_position == 5
        elif q_position in {24, 10, 19}:
            return k_position == 17
        elif q_position in {11}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 20
        elif q_position in {29, 13}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 21
        elif q_position in {16, 18, 38}:
            return k_position == 9
        elif q_position in {17}:
            return k_position == 13
        elif q_position in {25, 20}:
            return k_position == 18
        elif q_position in {49, 21}:
            return k_position == 16
        elif q_position in {23}:
            return k_position == 11
        elif q_position in {28}:
            return k_position == 22
        elif q_position in {30}:
            return k_position == 31
        elif q_position in {31}:
            return k_position == 12
        elif q_position in {32, 43, 45}:
            return k_position == 0
        elif q_position in {33}:
            return k_position == 28
        elif q_position in {34}:
            return k_position == 29
        elif q_position in {35}:
            return k_position == 33
        elif q_position in {42}:
            return k_position == 46
        elif q_position in {44, 47}:
            return k_position == 7

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_5_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(num_mlp_0_3_output, mlp_0_0_output):
        if num_mlp_0_3_output in {0, 35, 6, 8, 22, 26}:
            return mlp_0_0_output == 5
        elif num_mlp_0_3_output in {1, 33, 31}:
            return mlp_0_0_output == 39
        elif num_mlp_0_3_output in {2, 27, 39}:
            return mlp_0_0_output == 7
        elif num_mlp_0_3_output in {24, 9, 3}:
            return mlp_0_0_output == 45
        elif num_mlp_0_3_output in {34, 4, 43, 49, 19}:
            return mlp_0_0_output == 3
        elif num_mlp_0_3_output in {32, 28, 5, 23}:
            return mlp_0_0_output == 6
        elif num_mlp_0_3_output in {7}:
            return mlp_0_0_output == 0
        elif num_mlp_0_3_output in {10, 14}:
            return mlp_0_0_output == 19
        elif num_mlp_0_3_output in {11}:
            return mlp_0_0_output == 21
        elif num_mlp_0_3_output in {12}:
            return mlp_0_0_output == 23
        elif num_mlp_0_3_output in {48, 13}:
            return mlp_0_0_output == 38
        elif num_mlp_0_3_output in {15}:
            return mlp_0_0_output == 43
        elif num_mlp_0_3_output in {16}:
            return mlp_0_0_output == 44
        elif num_mlp_0_3_output in {40, 17, 18, 21}:
            return mlp_0_0_output == 2
        elif num_mlp_0_3_output in {41, 45, 46, 47, 20, 29}:
            return mlp_0_0_output == 13
        elif num_mlp_0_3_output in {25}:
            return mlp_0_0_output == 1
        elif num_mlp_0_3_output in {37, 30}:
            return mlp_0_0_output == 4
        elif num_mlp_0_3_output in {36}:
            return mlp_0_0_output == 14
        elif num_mlp_0_3_output in {38}:
            return mlp_0_0_output == 28
        elif num_mlp_0_3_output in {42}:
            return mlp_0_0_output == 9
        elif num_mlp_0_3_output in {44}:
            return mlp_0_0_output == 12

    attn_1_7_pattern = select_closest(
        mlp_0_0_outputs, num_mlp_0_3_outputs, predicate_1_7
    )
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_0_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            4,
            5,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            38,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif mlp_0_0_output in {1, 2, 35}:
            return token == "<s>"
        elif mlp_0_0_output in {11, 3}:
            return token == "</s>"
        elif mlp_0_0_output in {6}:
            return token == "0"
        elif mlp_0_0_output in {27, 37}:
            return token == "<pad>"
        elif mlp_0_0_output in {39}:
            return token == "2"

    num_attn_1_0_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_7_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            14,
            17,
            30,
            31,
            32,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == "0"
        elif mlp_0_0_output in {
            35,
            13,
            15,
            16,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
        }:
            return token == ""
        elif mlp_0_0_output in {25, 34}:
            return token == "</s>"
        elif mlp_0_0_output in {33}:
            return token == "2"
        elif mlp_0_0_output in {43}:
            return token == "1"

    num_attn_1_1_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_7_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {
            0,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {1, 2, 3, 35}:
            return token == "0"
        elif position in {25}:
            return token == "<s>"
        elif position in {26, 27}:
            return token == "</s>"
        elif position in {32, 28, 29, 30, 31}:
            return token == "4"
        elif position in {33, 34}:
            return token == "1"
        elif position in {36, 37, 38}:
            return token == "2"
        elif position in {45}:
            return token == "<pad>"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_2_output):
        if position in {
            0,
            5,
            7,
            18,
            21,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            41,
            42,
            43,
            44,
            45,
            46,
            48,
        }:
            return attn_0_2_output == ""
        elif position in {1, 2, 49}:
            return attn_0_2_output == "<s>"
        elif position in {3, 4, 19, 20, 22}:
            return attn_0_2_output == "</s>"
        elif position in {6, 8, 9, 10, 11, 12, 13, 15, 47}:
            return attn_0_2_output == "2"
        elif position in {40, 14}:
            return attn_0_2_output == "1"
        elif position in {16, 17}:
            return attn_0_2_output == "0"

    num_attn_1_3_pattern = select(attn_0_2_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_2_output):
        if position in {
            0,
            5,
            13,
            14,
            16,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            44,
        }:
            return attn_0_2_output == ""
        elif position in {1}:
            return attn_0_2_output == "2"
        elif position in {2, 42}:
            return attn_0_2_output == "1"
        elif position in {3, 6, 7, 8, 9, 10, 11, 39, 40, 41, 43, 45, 46, 47, 48, 49}:
            return attn_0_2_output == "0"
        elif position in {4, 12, 15, 17, 18, 21}:
            return attn_0_2_output == "</s>"
        elif position in {19, 20}:
            return attn_0_2_output == "<s>"

    num_attn_1_4_pattern = select(attn_0_2_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_1_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(num_mlp_0_2_output, position):
        if num_mlp_0_2_output in {0}:
            return position == 6
        elif num_mlp_0_2_output in {1, 41}:
            return position == 35
        elif num_mlp_0_2_output in {24, 2, 22, 23}:
            return position == 31
        elif num_mlp_0_2_output in {3}:
            return position == 37
        elif num_mlp_0_2_output in {49, 4}:
            return position == 26
        elif num_mlp_0_2_output in {40, 25, 5}:
            return position == 20
        elif num_mlp_0_2_output in {6}:
            return position == 15
        elif num_mlp_0_2_output in {32, 34, 36, 38, 7, 43, 48, 20, 27, 28}:
            return position == 0
        elif num_mlp_0_2_output in {8}:
            return position == 17
        elif num_mlp_0_2_output in {9, 21, 14}:
            return position == 23
        elif num_mlp_0_2_output in {10}:
            return position == 3
        elif num_mlp_0_2_output in {11, 45}:
            return position == 16
        elif num_mlp_0_2_output in {16, 12}:
            return position == 19
        elif num_mlp_0_2_output in {26, 19, 13}:
            return position == 34
        elif num_mlp_0_2_output in {15}:
            return position == 27
        elif num_mlp_0_2_output in {17}:
            return position == 32
        elif num_mlp_0_2_output in {18, 46}:
            return position == 36
        elif num_mlp_0_2_output in {29, 47}:
            return position == 28
        elif num_mlp_0_2_output in {30}:
            return position == 38
        elif num_mlp_0_2_output in {31}:
            return position == 12
        elif num_mlp_0_2_output in {33}:
            return position == 11
        elif num_mlp_0_2_output in {35}:
            return position == 18
        elif num_mlp_0_2_output in {42, 37}:
            return position == 24
        elif num_mlp_0_2_output in {39}:
            return position == 7
        elif num_mlp_0_2_output in {44}:
            return position == 44

    num_attn_1_5_pattern = select(positions, num_mlp_0_2_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_1_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_3_output):
        if position in {
            0,
            11,
            12,
            13,
            14,
            15,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return attn_0_3_output == ""
        elif position in {1, 2, 39}:
            return attn_0_3_output == "<s>"
        elif position in {16, 18, 3}:
            return attn_0_3_output == "</s>"
        elif position in {4, 5, 6, 7, 9, 10}:
            return attn_0_3_output == "1"
        elif position in {8}:
            return attn_0_3_output == "0"

    num_attn_1_6_pattern = select(attn_0_3_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_6_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 8, 41, 9, 43, 44, 15, 47}:
            return token == "0"
        elif mlp_0_0_output in {1, 49, 6, 39}:
            return token == "1"
        elif mlp_0_0_output in {2, 3}:
            return token == "</s>"
        elif mlp_0_0_output in {
            4,
            5,
            7,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            40,
            42,
            45,
            46,
            48,
        }:
            return token == ""

    num_attn_1_7_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_7_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output, attn_1_5_output):
        key = (attn_1_0_output, attn_1_5_output)
        if key in {("1", "4"), ("3", "3"), ("3", "4"), ("4", "3"), ("4", "4")}:
            return 41
        return 33

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(mlp_0_3_output, num_mlp_0_1_output):
        key = (mlp_0_3_output, num_mlp_0_1_output)
        if key in {
            (0, 8),
            (0, 17),
            (1, 8),
            (1, 17),
            (1, 46),
            (2, 8),
            (2, 17),
            (3, 8),
            (3, 17),
            (4, 8),
            (4, 17),
            (6, 8),
            (6, 17),
            (9, 8),
            (9, 17),
            (9, 46),
            (11, 8),
            (11, 17),
            (12, 8),
            (12, 17),
            (12, 46),
            (14, 8),
            (14, 17),
            (16, 8),
            (16, 17),
            (16, 46),
            (21, 8),
            (21, 17),
            (21, 46),
            (26, 8),
            (26, 17),
            (28, 8),
            (28, 17),
            (28, 46),
            (32, 8),
            (32, 11),
            (32, 17),
            (32, 46),
            (34, 8),
            (34, 17),
            (34, 46),
            (36, 8),
            (36, 17),
            (36, 46),
            (39, 8),
            (39, 17),
            (40, 8),
            (40, 17),
            (40, 46),
            (42, 8),
            (42, 17),
            (46, 8),
            (46, 17),
            (46, 46),
            (47, 8),
            (47, 17),
            (47, 46),
            (48, 8),
            (48, 17),
        }:
            return 47
        return 10

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(mlp_0_3_outputs, num_mlp_0_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(num_mlp_0_3_output, num_mlp_0_2_output):
        key = (num_mlp_0_3_output, num_mlp_0_2_output)
        if key in {
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 20),
            (1, 25),
            (1, 46),
            (2, 4),
            (2, 5),
            (2, 20),
            (2, 25),
            (15, 4),
            (15, 5),
            (15, 8),
            (15, 20),
            (15, 25),
            (24, 4),
            (24, 5),
            (24, 8),
            (24, 20),
            (24, 25),
            (27, 4),
            (27, 5),
            (27, 20),
            (27, 25),
            (28, 4),
            (28, 5),
            (28, 8),
            (28, 16),
            (28, 18),
            (28, 20),
            (28, 24),
            (28, 25),
            (28, 46),
            (33, 4),
            (33, 5),
            (33, 8),
            (33, 11),
            (33, 14),
            (33, 16),
            (33, 18),
            (33, 20),
            (33, 24),
            (33, 25),
            (33, 46),
            (36, 5),
            (36, 25),
            (41, 4),
            (41, 5),
            (41, 8),
            (41, 20),
            (41, 25),
            (43, 4),
            (43, 5),
            (43, 8),
            (43, 20),
            (43, 25),
            (44, 5),
            (44, 25),
        }:
            return 28
        return 8

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, num_mlp_0_2_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(position, num_mlp_0_0_output):
        key = (position, num_mlp_0_0_output)
        if key in {
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 48),
            (2, 49),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 29),
            (10, 30),
            (10, 31),
            (10, 32),
            (10, 33),
            (10, 34),
            (10, 35),
            (10, 36),
            (10, 37),
            (10, 38),
            (10, 39),
            (10, 40),
            (10, 41),
            (10, 42),
            (10, 43),
            (10, 44),
            (10, 45),
            (10, 46),
            (10, 47),
            (10, 48),
            (10, 49),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 14),
            (15, 15),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 19),
            (15, 20),
            (15, 21),
            (15, 22),
            (15, 23),
            (15, 24),
            (15, 25),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 29),
            (15, 30),
            (15, 31),
            (15, 32),
            (15, 33),
            (15, 34),
            (15, 35),
            (15, 36),
            (15, 37),
            (15, 38),
            (15, 39),
            (15, 40),
            (15, 41),
            (15, 42),
            (15, 43),
            (15, 44),
            (15, 45),
            (15, 46),
            (15, 47),
            (15, 48),
            (15, 49),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (21, 11),
            (21, 12),
            (21, 13),
            (21, 14),
            (21, 15),
            (21, 16),
            (21, 17),
            (21, 18),
            (21, 19),
            (21, 20),
            (21, 21),
            (21, 22),
            (21, 23),
            (21, 24),
            (21, 25),
            (21, 26),
            (21, 27),
            (21, 28),
            (21, 29),
            (21, 31),
            (21, 34),
            (21, 35),
            (21, 36),
            (21, 37),
            (21, 38),
            (21, 39),
            (21, 40),
            (21, 41),
            (21, 42),
            (21, 43),
            (21, 44),
            (21, 45),
            (21, 46),
            (21, 47),
            (21, 49),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 11),
            (22, 12),
            (22, 13),
            (22, 14),
            (22, 15),
            (22, 16),
            (22, 17),
            (22, 18),
            (22, 19),
            (22, 20),
            (22, 21),
            (22, 22),
            (22, 23),
            (22, 24),
            (22, 25),
            (22, 26),
            (22, 27),
            (22, 28),
            (22, 29),
            (22, 31),
            (22, 34),
            (22, 35),
            (22, 36),
            (22, 37),
            (22, 38),
            (22, 39),
            (22, 40),
            (22, 41),
            (22, 42),
            (22, 43),
            (22, 44),
            (22, 45),
            (22, 46),
            (22, 47),
            (22, 49),
        }:
            return 8
        elif key in {
            (18, 23),
            (20, 23),
            (21, 10),
            (21, 30),
            (21, 32),
            (21, 33),
            (21, 48),
            (22, 0),
            (22, 10),
            (22, 30),
            (22, 32),
            (22, 33),
            (22, 48),
            (35, 23),
            (35, 31),
            (40, 23),
        }:
            return 19
        return 40

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(positions, num_mlp_0_0_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output, num_attn_1_5_output):
        key = (num_attn_1_7_output, num_attn_1_5_output)
        return 7

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_1_0_output):
        key = (num_attn_1_2_output, num_attn_1_0_output)
        return 10

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_1_output, num_attn_1_5_output):
        key = (num_attn_1_1_output, num_attn_1_5_output)
        return 12

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_1_output, num_attn_1_0_output):
        key = (num_attn_1_1_output, num_attn_1_0_output)
        return 30

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"0", "<s>", "4"}:
            return k_token == ""
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "2"
        elif q_token in {"</s>"}:
            return k_token == "4"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_6_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, attn_1_3_output):
        if token in {"0"}:
            return attn_1_3_output == "2"
        elif token in {"1"}:
            return attn_1_3_output == "3"
        elif token in {"2"}:
            return attn_1_3_output == "1"
        elif token in {"3"}:
            return attn_1_3_output == "<pad>"
        elif token in {"</s>", "4"}:
            return attn_1_3_output == "<s>"
        elif token in {"<s>"}:
            return attn_1_3_output == "</s>"

    attn_2_1_pattern = select_closest(attn_1_3_outputs, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_7_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0", "2", "3"}:
            return k_token == "4"
        elif q_token in {"1", "<s>"}:
            return k_token == ""
        elif q_token in {"4"}:
            return k_token == "1"
        elif q_token in {"</s>"}:
            return k_token == "<s>"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_6_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_0_output, attn_0_0_output):
        if mlp_0_0_output in {0}:
            return attn_0_0_output == "<pad>"
        elif mlp_0_0_output in {1, 40, 41, 44, 45, 47, 49, 26, 27}:
            return attn_0_0_output == "2"
        elif mlp_0_0_output in {2, 13, 6, 7}:
            return attn_0_0_output == "<s>"
        elif mlp_0_0_output in {
            3,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            24,
            28,
            30,
            32,
            34,
            35,
            36,
            37,
            42,
            43,
            46,
            48,
        }:
            return attn_0_0_output == ""
        elif mlp_0_0_output in {10, 11, 4}:
            return attn_0_0_output == "1"
        elif mlp_0_0_output in {25, 12, 5}:
            return attn_0_0_output == "</s>"
        elif mlp_0_0_output in {8, 38}:
            return attn_0_0_output == "3"
        elif mlp_0_0_output in {9, 23}:
            return attn_0_0_output == "4"
        elif mlp_0_0_output in {33, 39, 14, 15, 29, 31}:
            return attn_0_0_output == "0"

    attn_2_3_pattern = select_closest(attn_0_0_outputs, mlp_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_token, k_token):
        if q_token in {"0", "1", "3"}:
            return k_token == "4"
        elif q_token in {"2"}:
            return k_token == "0"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"</s>", "<s>"}:
            return k_token == "<pad>"

    attn_2_4_pattern = select_closest(tokens, tokens, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_4_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(token, num_mlp_1_0_output):
        if token in {"0"}:
            return num_mlp_1_0_output == 36
        elif token in {"1", "3"}:
            return num_mlp_1_0_output == 0
        elif token in {"2"}:
            return num_mlp_1_0_output == 32
        elif token in {"4"}:
            return num_mlp_1_0_output == 27
        elif token in {"</s>", "<s>"}:
            return num_mlp_1_0_output == 7

    attn_2_5_pattern = select_closest(num_mlp_1_0_outputs, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_0_7_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"0", "3"}:
            return k_token == "2"
        elif q_token in {"1", "</s>", "2", "4"}:
            return k_token == ""
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_4_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_position, k_position):
        if q_position in {0, 32, 34, 36, 37, 38, 43, 48}:
            return k_position == 0
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 3}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 12
        elif q_position in {5}:
            return k_position == 14
        elif q_position in {44, 6}:
            return k_position == 10
        elif q_position in {8, 9, 7}:
            return k_position == 18
        elif q_position in {10, 11, 14, 15}:
            return k_position == 21
        elif q_position in {12}:
            return k_position == 20
        elif q_position in {13}:
            return k_position == 28
        elif q_position in {16, 18, 23}:
            return k_position == 37
        elif q_position in {17, 49}:
            return k_position == 13
        elif q_position in {19}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 16
        elif q_position in {21, 22}:
            return k_position == 36
        elif q_position in {24, 25}:
            return k_position == 38
        elif q_position in {26, 30}:
            return k_position == 39
        elif q_position in {27}:
            return k_position == 7
        elif q_position in {42, 28}:
            return k_position == 5
        elif q_position in {29}:
            return k_position == 8
        elif q_position in {31}:
            return k_position == 34
        elif q_position in {33}:
            return k_position == 32
        elif q_position in {35}:
            return k_position == 31
        elif q_position in {39}:
            return k_position == 1
        elif q_position in {40}:
            return k_position == 15
        elif q_position in {41}:
            return k_position == 30
        elif q_position in {45}:
            return k_position == 4
        elif q_position in {46}:
            return k_position == 40
        elif q_position in {47}:
            return k_position == 48

    attn_2_7_pattern = select_closest(positions, positions, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_6_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_5_output, token):
        if attn_1_5_output in {"3", "1", "</s>", "<s>", "0", "4"}:
            return token == ""
        elif attn_1_5_output in {"2"}:
            return token == "0"

    num_attn_2_0_pattern = select(tokens, attn_1_5_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_3_output, mlp_0_1_output):
        if attn_1_3_output in {"0"}:
            return mlp_0_1_output == 6
        elif attn_1_3_output in {"1"}:
            return mlp_0_1_output == 31
        elif attn_1_3_output in {"2"}:
            return mlp_0_1_output == 8
        elif attn_1_3_output in {"</s>", "3"}:
            return mlp_0_1_output == 24
        elif attn_1_3_output in {"4"}:
            return mlp_0_1_output == 20
        elif attn_1_3_output in {"<s>"}:
            return mlp_0_1_output == 25

    num_attn_2_1_pattern = select(mlp_0_1_outputs, attn_1_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_5_output, attn_1_1_output):
        if attn_1_5_output in {"2", "1", "<s>", "0", "4"}:
            return attn_1_1_output == ""
        elif attn_1_5_output in {"3"}:
            return attn_1_1_output == "1"
        elif attn_1_5_output in {"</s>"}:
            return attn_1_1_output == "</s>"

    num_attn_2_2_pattern = select(attn_1_1_outputs, attn_1_5_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_5_output, attn_1_0_output):
        if attn_1_5_output in {"0", "1", "</s>", "<s>"}:
            return attn_1_0_output == "0"
        elif attn_1_5_output in {"2"}:
            return attn_1_0_output == "<s>"
        elif attn_1_5_output in {"3", "4"}:
            return attn_1_0_output == ""

    num_attn_2_3_pattern = select(attn_1_0_outputs, attn_1_5_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_2_output, token):
        if attn_1_2_output in {"0", "3", "4"}:
            return token == ""
        elif attn_1_2_output in {"1", "<s>"}:
            return token == "2"
        elif attn_1_2_output in {"2"}:
            return token == "</s>"
        elif attn_1_2_output in {"</s>"}:
            return token == "<s>"

    num_attn_2_4_pattern = select(tokens, attn_1_2_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_3_output, num_mlp_0_3_output):
        if attn_1_3_output in {"0"}:
            return num_mlp_0_3_output == 35
        elif attn_1_3_output in {"1"}:
            return num_mlp_0_3_output == 32
        elif attn_1_3_output in {"2"}:
            return num_mlp_0_3_output == 21
        elif attn_1_3_output in {"3"}:
            return num_mlp_0_3_output == 8
        elif attn_1_3_output in {"4"}:
            return num_mlp_0_3_output == 36
        elif attn_1_3_output in {"</s>"}:
            return num_mlp_0_3_output == 18
        elif attn_1_3_output in {"<s>"}:
            return num_mlp_0_3_output == 27

    num_attn_2_5_pattern = select(
        num_mlp_0_3_outputs, attn_1_3_outputs, num_predicate_2_5
    )
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_7_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(position, attn_1_1_output):
        if position in {
            0,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return attn_1_1_output == ""
        elif position in {1, 2, 3, 4}:
            return attn_1_1_output == "2"
        elif position in {5}:
            return attn_1_1_output == "3"
        elif position in {6, 7, 8, 9, 10, 11, 12, 13}:
            return attn_1_1_output == "1"
        elif position in {14}:
            return attn_1_1_output == "<s>"
        elif position in {15}:
            return attn_1_1_output == "0"
        elif position in {16, 17, 18, 19}:
            return attn_1_1_output == "</s>"

    num_attn_2_6_pattern = select(attn_1_1_outputs, positions, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_6_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(position, attn_1_0_output):
        if position in {
            0,
            1,
            3,
            4,
            7,
            8,
            9,
            10,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
        }:
            return attn_1_0_output == "0"
        elif position in {2, 11, 21}:
            return attn_1_0_output == "<s>"
        elif position in {
            5,
            13,
            14,
            15,
            17,
            18,
            19,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            45,
        }:
            return attn_1_0_output == ""
        elif position in {6}:
            return attn_1_0_output == "1"
        elif position in {16, 12, 20, 22}:
            return attn_1_0_output == "</s>"

    num_attn_2_7_pattern = select(attn_1_0_outputs, positions, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_1_output, mlp_0_1_output):
        key = (attn_0_1_output, mlp_0_1_output)
        if key in {
            ("</s>", 2),
            ("</s>", 22),
            ("</s>", 32),
            ("<s>", 2),
            ("<s>", 22),
            ("<s>", 32),
        }:
            return 47
        return 20

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, mlp_0_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position, attn_2_3_output):
        key = (position, attn_2_3_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (39, "1"),
        }:
            return 40
        elif key in {
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "</s>"),
            (3, "<s>"),
            (39, "0"),
        }:
            return 12
        elif key in {
            (0, "0"),
            (15, "0"),
            (18, "0"),
            (19, "0"),
            (20, "0"),
            (23, "0"),
            (40, "0"),
            (41, "0"),
            (42, "0"),
            (43, "0"),
            (44, "0"),
            (45, "0"),
            (46, "0"),
            (47, "0"),
            (48, "0"),
            (49, "0"),
        }:
            return 30
        return 1

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(positions, attn_2_3_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(num_mlp_0_1_output, attn_2_1_output):
        key = (num_mlp_0_1_output, attn_2_1_output)
        return 34

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_2_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_0_2_output, mlp_1_0_output):
        key = (num_mlp_0_2_output, mlp_1_0_output)
        if key in {(11, 41), (11, 43), (13, 18), (13, 41), (13, 43)}:
            return 1
        return 16

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_0_2_outputs, mlp_1_0_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_0_2_output):
        key = (num_attn_1_2_output, num_attn_0_2_output)
        if key in {
            (0, 0),
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (5, 2),
            (6, 0),
            (6, 1),
            (6, 2),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 9),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (23, 10),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (24, 9),
            (24, 10),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 9),
            (25, 10),
            (25, 11),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 9),
            (26, 10),
            (26, 11),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 10),
            (27, 11),
            (27, 12),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 11),
            (28, 12),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 10),
            (29, 11),
            (29, 12),
            (29, 13),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 10),
            (30, 11),
            (30, 12),
            (30, 13),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (31, 8),
            (31, 9),
            (31, 10),
            (31, 11),
            (31, 12),
            (31, 13),
            (31, 14),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (32, 7),
            (32, 8),
            (32, 9),
            (32, 10),
            (32, 11),
            (32, 12),
            (32, 13),
            (32, 14),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (33, 8),
            (33, 9),
            (33, 10),
            (33, 11),
            (33, 12),
            (33, 13),
            (33, 14),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (34, 9),
            (34, 10),
            (34, 11),
            (34, 12),
            (34, 13),
            (34, 14),
            (34, 15),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 8),
            (35, 9),
            (35, 10),
            (35, 11),
            (35, 12),
            (35, 13),
            (35, 14),
            (35, 15),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 8),
            (36, 9),
            (36, 10),
            (36, 11),
            (36, 12),
            (36, 13),
            (36, 14),
            (36, 15),
            (36, 16),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 8),
            (37, 9),
            (37, 10),
            (37, 11),
            (37, 12),
            (37, 13),
            (37, 14),
            (37, 15),
            (37, 16),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (38, 8),
            (38, 9),
            (38, 10),
            (38, 11),
            (38, 12),
            (38, 13),
            (38, 14),
            (38, 15),
            (38, 16),
            (38, 17),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 9),
            (39, 10),
            (39, 11),
            (39, 12),
            (39, 13),
            (39, 14),
            (39, 15),
            (39, 16),
            (39, 17),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (40, 9),
            (40, 10),
            (40, 11),
            (40, 12),
            (40, 13),
            (40, 14),
            (40, 15),
            (40, 16),
            (40, 17),
            (40, 18),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (41, 9),
            (41, 10),
            (41, 11),
            (41, 12),
            (41, 13),
            (41, 14),
            (41, 15),
            (41, 16),
            (41, 17),
            (41, 18),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (42, 9),
            (42, 10),
            (42, 11),
            (42, 12),
            (42, 13),
            (42, 14),
            (42, 15),
            (42, 16),
            (42, 17),
            (42, 18),
            (42, 19),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (43, 9),
            (43, 10),
            (43, 11),
            (43, 12),
            (43, 13),
            (43, 14),
            (43, 15),
            (43, 16),
            (43, 17),
            (43, 18),
            (43, 19),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (44, 9),
            (44, 10),
            (44, 11),
            (44, 12),
            (44, 13),
            (44, 14),
            (44, 15),
            (44, 16),
            (44, 17),
            (44, 18),
            (44, 19),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (45, 9),
            (45, 10),
            (45, 11),
            (45, 12),
            (45, 13),
            (45, 14),
            (45, 15),
            (45, 16),
            (45, 17),
            (45, 18),
            (45, 19),
            (45, 20),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (46, 9),
            (46, 10),
            (46, 11),
            (46, 12),
            (46, 13),
            (46, 14),
            (46, 15),
            (46, 16),
            (46, 17),
            (46, 18),
            (46, 19),
            (46, 20),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
            (47, 9),
            (47, 10),
            (47, 11),
            (47, 12),
            (47, 13),
            (47, 14),
            (47, 15),
            (47, 16),
            (47, 17),
            (47, 18),
            (47, 19),
            (47, 20),
            (47, 21),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (48, 5),
            (48, 6),
            (48, 7),
            (48, 8),
            (48, 9),
            (48, 10),
            (48, 11),
            (48, 12),
            (48, 13),
            (48, 14),
            (48, 15),
            (48, 16),
            (48, 17),
            (48, 18),
            (48, 19),
            (48, 20),
            (48, 21),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (49, 5),
            (49, 6),
            (49, 7),
            (49, 8),
            (49, 9),
            (49, 10),
            (49, 11),
            (49, 12),
            (49, 13),
            (49, 14),
            (49, 15),
            (49, 16),
            (49, 17),
            (49, 18),
            (49, 19),
            (49, 20),
            (49, 21),
            (49, 22),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (50, 5),
            (50, 6),
            (50, 7),
            (50, 8),
            (50, 9),
            (50, 10),
            (50, 11),
            (50, 12),
            (50, 13),
            (50, 14),
            (50, 15),
            (50, 16),
            (50, 17),
            (50, 18),
            (50, 19),
            (50, 20),
            (50, 21),
            (50, 22),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (51, 5),
            (51, 6),
            (51, 7),
            (51, 8),
            (51, 9),
            (51, 10),
            (51, 11),
            (51, 12),
            (51, 13),
            (51, 14),
            (51, 15),
            (51, 16),
            (51, 17),
            (51, 18),
            (51, 19),
            (51, 20),
            (51, 21),
            (51, 22),
            (51, 23),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (52, 5),
            (52, 6),
            (52, 7),
            (52, 8),
            (52, 9),
            (52, 10),
            (52, 11),
            (52, 12),
            (52, 13),
            (52, 14),
            (52, 15),
            (52, 16),
            (52, 17),
            (52, 18),
            (52, 19),
            (52, 20),
            (52, 21),
            (52, 22),
            (52, 23),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (53, 5),
            (53, 6),
            (53, 7),
            (53, 8),
            (53, 9),
            (53, 10),
            (53, 11),
            (53, 12),
            (53, 13),
            (53, 14),
            (53, 15),
            (53, 16),
            (53, 17),
            (53, 18),
            (53, 19),
            (53, 20),
            (53, 21),
            (53, 22),
            (53, 23),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (54, 5),
            (54, 6),
            (54, 7),
            (54, 8),
            (54, 9),
            (54, 10),
            (54, 11),
            (54, 12),
            (54, 13),
            (54, 14),
            (54, 15),
            (54, 16),
            (54, 17),
            (54, 18),
            (54, 19),
            (54, 20),
            (54, 21),
            (54, 22),
            (54, 23),
            (54, 24),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (55, 5),
            (55, 6),
            (55, 7),
            (55, 8),
            (55, 9),
            (55, 10),
            (55, 11),
            (55, 12),
            (55, 13),
            (55, 14),
            (55, 15),
            (55, 16),
            (55, 17),
            (55, 18),
            (55, 19),
            (55, 20),
            (55, 21),
            (55, 22),
            (55, 23),
            (55, 24),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (56, 5),
            (56, 6),
            (56, 7),
            (56, 8),
            (56, 9),
            (56, 10),
            (56, 11),
            (56, 12),
            (56, 13),
            (56, 14),
            (56, 15),
            (56, 16),
            (56, 17),
            (56, 18),
            (56, 19),
            (56, 20),
            (56, 21),
            (56, 22),
            (56, 23),
            (56, 24),
            (56, 25),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (57, 6),
            (57, 7),
            (57, 8),
            (57, 9),
            (57, 10),
            (57, 11),
            (57, 12),
            (57, 13),
            (57, 14),
            (57, 15),
            (57, 16),
            (57, 17),
            (57, 18),
            (57, 19),
            (57, 20),
            (57, 21),
            (57, 22),
            (57, 23),
            (57, 24),
            (57, 25),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (58, 6),
            (58, 7),
            (58, 8),
            (58, 9),
            (58, 10),
            (58, 11),
            (58, 12),
            (58, 13),
            (58, 14),
            (58, 15),
            (58, 16),
            (58, 17),
            (58, 18),
            (58, 19),
            (58, 20),
            (58, 21),
            (58, 22),
            (58, 23),
            (58, 24),
            (58, 25),
            (58, 26),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (59, 6),
            (59, 7),
            (59, 8),
            (59, 9),
            (59, 10),
            (59, 11),
            (59, 12),
            (59, 13),
            (59, 14),
            (59, 15),
            (59, 16),
            (59, 17),
            (59, 18),
            (59, 19),
            (59, 20),
            (59, 21),
            (59, 22),
            (59, 23),
            (59, 24),
            (59, 25),
            (59, 26),
            (60, 0),
            (60, 1),
            (60, 2),
            (60, 3),
            (60, 4),
            (60, 5),
            (60, 6),
            (60, 7),
            (60, 8),
            (60, 9),
            (60, 10),
            (60, 11),
            (60, 12),
            (60, 13),
            (60, 14),
            (60, 15),
            (60, 16),
            (60, 17),
            (60, 18),
            (60, 19),
            (60, 20),
            (60, 21),
            (60, 22),
            (60, 23),
            (60, 24),
            (60, 25),
            (60, 26),
            (60, 27),
            (61, 0),
            (61, 1),
            (61, 2),
            (61, 3),
            (61, 4),
            (61, 5),
            (61, 6),
            (61, 7),
            (61, 8),
            (61, 9),
            (61, 10),
            (61, 11),
            (61, 12),
            (61, 13),
            (61, 14),
            (61, 15),
            (61, 16),
            (61, 17),
            (61, 18),
            (61, 19),
            (61, 20),
            (61, 21),
            (61, 22),
            (61, 23),
            (61, 24),
            (61, 25),
            (61, 26),
            (61, 27),
            (62, 0),
            (62, 1),
            (62, 2),
            (62, 3),
            (62, 4),
            (62, 5),
            (62, 6),
            (62, 7),
            (62, 8),
            (62, 9),
            (62, 10),
            (62, 11),
            (62, 12),
            (62, 13),
            (62, 14),
            (62, 15),
            (62, 16),
            (62, 17),
            (62, 18),
            (62, 19),
            (62, 20),
            (62, 21),
            (62, 22),
            (62, 23),
            (62, 24),
            (62, 25),
            (62, 26),
            (62, 27),
            (62, 28),
            (63, 0),
            (63, 1),
            (63, 2),
            (63, 3),
            (63, 4),
            (63, 5),
            (63, 6),
            (63, 7),
            (63, 8),
            (63, 9),
            (63, 10),
            (63, 11),
            (63, 12),
            (63, 13),
            (63, 14),
            (63, 15),
            (63, 16),
            (63, 17),
            (63, 18),
            (63, 19),
            (63, 20),
            (63, 21),
            (63, 22),
            (63, 23),
            (63, 24),
            (63, 25),
            (63, 26),
            (63, 27),
            (63, 28),
            (64, 0),
            (64, 1),
            (64, 2),
            (64, 3),
            (64, 4),
            (64, 5),
            (64, 6),
            (64, 7),
            (64, 8),
            (64, 9),
            (64, 10),
            (64, 11),
            (64, 12),
            (64, 13),
            (64, 14),
            (64, 15),
            (64, 16),
            (64, 17),
            (64, 18),
            (64, 19),
            (64, 20),
            (64, 21),
            (64, 22),
            (64, 23),
            (64, 24),
            (64, 25),
            (64, 26),
            (64, 27),
            (64, 28),
            (65, 0),
            (65, 1),
            (65, 2),
            (65, 3),
            (65, 4),
            (65, 5),
            (65, 6),
            (65, 7),
            (65, 8),
            (65, 9),
            (65, 10),
            (65, 11),
            (65, 12),
            (65, 13),
            (65, 14),
            (65, 15),
            (65, 16),
            (65, 17),
            (65, 18),
            (65, 19),
            (65, 20),
            (65, 21),
            (65, 22),
            (65, 23),
            (65, 24),
            (65, 25),
            (65, 26),
            (65, 27),
            (65, 28),
            (65, 29),
            (66, 0),
            (66, 1),
            (66, 2),
            (66, 3),
            (66, 4),
            (66, 5),
            (66, 6),
            (66, 7),
            (66, 8),
            (66, 9),
            (66, 10),
            (66, 11),
            (66, 12),
            (66, 13),
            (66, 14),
            (66, 15),
            (66, 16),
            (66, 17),
            (66, 18),
            (66, 19),
            (66, 20),
            (66, 21),
            (66, 22),
            (66, 23),
            (66, 24),
            (66, 25),
            (66, 26),
            (66, 27),
            (66, 28),
            (66, 29),
            (67, 0),
            (67, 1),
            (67, 2),
            (67, 3),
            (67, 4),
            (67, 5),
            (67, 6),
            (67, 7),
            (67, 8),
            (67, 9),
            (67, 10),
            (67, 11),
            (67, 12),
            (67, 13),
            (67, 14),
            (67, 15),
            (67, 16),
            (67, 17),
            (67, 18),
            (67, 19),
            (67, 20),
            (67, 21),
            (67, 22),
            (67, 23),
            (67, 24),
            (67, 25),
            (67, 26),
            (67, 27),
            (67, 28),
            (67, 29),
            (67, 30),
            (68, 0),
            (68, 1),
            (68, 2),
            (68, 3),
            (68, 4),
            (68, 5),
            (68, 6),
            (68, 7),
            (68, 8),
            (68, 9),
            (68, 10),
            (68, 11),
            (68, 12),
            (68, 13),
            (68, 14),
            (68, 15),
            (68, 16),
            (68, 17),
            (68, 18),
            (68, 19),
            (68, 20),
            (68, 21),
            (68, 22),
            (68, 23),
            (68, 24),
            (68, 25),
            (68, 26),
            (68, 27),
            (68, 28),
            (68, 29),
            (68, 30),
            (69, 0),
            (69, 1),
            (69, 2),
            (69, 3),
            (69, 4),
            (69, 5),
            (69, 6),
            (69, 7),
            (69, 8),
            (69, 9),
            (69, 10),
            (69, 11),
            (69, 12),
            (69, 13),
            (69, 14),
            (69, 15),
            (69, 16),
            (69, 17),
            (69, 18),
            (69, 19),
            (69, 20),
            (69, 21),
            (69, 22),
            (69, 23),
            (69, 24),
            (69, 25),
            (69, 26),
            (69, 27),
            (69, 28),
            (69, 29),
            (69, 30),
            (69, 31),
            (70, 0),
            (70, 1),
            (70, 2),
            (70, 3),
            (70, 4),
            (70, 5),
            (70, 6),
            (70, 7),
            (70, 8),
            (70, 9),
            (70, 10),
            (70, 11),
            (70, 12),
            (70, 13),
            (70, 14),
            (70, 15),
            (70, 16),
            (70, 17),
            (70, 18),
            (70, 19),
            (70, 20),
            (70, 21),
            (70, 22),
            (70, 23),
            (70, 24),
            (70, 25),
            (70, 26),
            (70, 27),
            (70, 28),
            (70, 29),
            (70, 30),
            (70, 31),
            (71, 0),
            (71, 1),
            (71, 2),
            (71, 3),
            (71, 4),
            (71, 5),
            (71, 6),
            (71, 7),
            (71, 8),
            (71, 9),
            (71, 10),
            (71, 11),
            (71, 12),
            (71, 13),
            (71, 14),
            (71, 15),
            (71, 16),
            (71, 17),
            (71, 18),
            (71, 19),
            (71, 20),
            (71, 21),
            (71, 22),
            (71, 23),
            (71, 24),
            (71, 25),
            (71, 26),
            (71, 27),
            (71, 28),
            (71, 29),
            (71, 30),
            (71, 31),
            (71, 32),
            (72, 0),
            (72, 1),
            (72, 2),
            (72, 3),
            (72, 4),
            (72, 5),
            (72, 6),
            (72, 7),
            (72, 8),
            (72, 9),
            (72, 10),
            (72, 11),
            (72, 12),
            (72, 13),
            (72, 14),
            (72, 15),
            (72, 16),
            (72, 17),
            (72, 18),
            (72, 19),
            (72, 20),
            (72, 21),
            (72, 22),
            (72, 23),
            (72, 24),
            (72, 25),
            (72, 26),
            (72, 27),
            (72, 28),
            (72, 29),
            (72, 30),
            (72, 31),
            (72, 32),
            (73, 0),
            (73, 1),
            (73, 2),
            (73, 3),
            (73, 4),
            (73, 5),
            (73, 6),
            (73, 7),
            (73, 8),
            (73, 9),
            (73, 10),
            (73, 11),
            (73, 12),
            (73, 13),
            (73, 14),
            (73, 15),
            (73, 16),
            (73, 17),
            (73, 18),
            (73, 19),
            (73, 20),
            (73, 21),
            (73, 22),
            (73, 23),
            (73, 24),
            (73, 25),
            (73, 26),
            (73, 27),
            (73, 28),
            (73, 29),
            (73, 30),
            (73, 31),
            (73, 32),
            (74, 0),
            (74, 1),
            (74, 2),
            (74, 3),
            (74, 4),
            (74, 5),
            (74, 6),
            (74, 7),
            (74, 8),
            (74, 9),
            (74, 10),
            (74, 11),
            (74, 12),
            (74, 13),
            (74, 14),
            (74, 15),
            (74, 16),
            (74, 17),
            (74, 18),
            (74, 19),
            (74, 20),
            (74, 21),
            (74, 22),
            (74, 23),
            (74, 24),
            (74, 25),
            (74, 26),
            (74, 27),
            (74, 28),
            (74, 29),
            (74, 30),
            (74, 31),
            (74, 32),
            (74, 33),
            (75, 0),
            (75, 1),
            (75, 2),
            (75, 3),
            (75, 4),
            (75, 5),
            (75, 6),
            (75, 7),
            (75, 8),
            (75, 9),
            (75, 10),
            (75, 11),
            (75, 12),
            (75, 13),
            (75, 14),
            (75, 15),
            (75, 16),
            (75, 17),
            (75, 18),
            (75, 19),
            (75, 20),
            (75, 21),
            (75, 22),
            (75, 23),
            (75, 24),
            (75, 25),
            (75, 26),
            (75, 27),
            (75, 28),
            (75, 29),
            (75, 30),
            (75, 31),
            (75, 32),
            (75, 33),
            (76, 0),
            (76, 1),
            (76, 2),
            (76, 3),
            (76, 4),
            (76, 5),
            (76, 6),
            (76, 7),
            (76, 8),
            (76, 9),
            (76, 10),
            (76, 11),
            (76, 12),
            (76, 13),
            (76, 14),
            (76, 15),
            (76, 16),
            (76, 17),
            (76, 18),
            (76, 19),
            (76, 20),
            (76, 21),
            (76, 22),
            (76, 23),
            (76, 24),
            (76, 25),
            (76, 26),
            (76, 27),
            (76, 28),
            (76, 29),
            (76, 30),
            (76, 31),
            (76, 32),
            (76, 33),
            (76, 34),
            (77, 0),
            (77, 1),
            (77, 2),
            (77, 3),
            (77, 4),
            (77, 5),
            (77, 6),
            (77, 7),
            (77, 8),
            (77, 9),
            (77, 10),
            (77, 11),
            (77, 12),
            (77, 13),
            (77, 14),
            (77, 15),
            (77, 16),
            (77, 17),
            (77, 18),
            (77, 19),
            (77, 20),
            (77, 21),
            (77, 22),
            (77, 23),
            (77, 24),
            (77, 25),
            (77, 26),
            (77, 27),
            (77, 28),
            (77, 29),
            (77, 30),
            (77, 31),
            (77, 32),
            (77, 33),
            (77, 34),
            (78, 0),
            (78, 1),
            (78, 2),
            (78, 3),
            (78, 4),
            (78, 5),
            (78, 6),
            (78, 7),
            (78, 8),
            (78, 9),
            (78, 10),
            (78, 11),
            (78, 12),
            (78, 13),
            (78, 14),
            (78, 15),
            (78, 16),
            (78, 17),
            (78, 18),
            (78, 19),
            (78, 20),
            (78, 21),
            (78, 22),
            (78, 23),
            (78, 24),
            (78, 25),
            (78, 26),
            (78, 27),
            (78, 28),
            (78, 29),
            (78, 30),
            (78, 31),
            (78, 32),
            (78, 33),
            (78, 34),
            (78, 35),
            (79, 0),
            (79, 1),
            (79, 2),
            (79, 3),
            (79, 4),
            (79, 5),
            (79, 6),
            (79, 7),
            (79, 8),
            (79, 9),
            (79, 10),
            (79, 11),
            (79, 12),
            (79, 13),
            (79, 14),
            (79, 15),
            (79, 16),
            (79, 17),
            (79, 18),
            (79, 19),
            (79, 20),
            (79, 21),
            (79, 22),
            (79, 23),
            (79, 24),
            (79, 25),
            (79, 26),
            (79, 27),
            (79, 28),
            (79, 29),
            (79, 30),
            (79, 31),
            (79, 32),
            (79, 33),
            (79, 34),
            (79, 35),
            (80, 0),
            (80, 1),
            (80, 2),
            (80, 3),
            (80, 4),
            (80, 5),
            (80, 6),
            (80, 7),
            (80, 8),
            (80, 9),
            (80, 10),
            (80, 11),
            (80, 12),
            (80, 13),
            (80, 14),
            (80, 15),
            (80, 16),
            (80, 17),
            (80, 18),
            (80, 19),
            (80, 20),
            (80, 21),
            (80, 22),
            (80, 23),
            (80, 24),
            (80, 25),
            (80, 26),
            (80, 27),
            (80, 28),
            (80, 29),
            (80, 30),
            (80, 31),
            (80, 32),
            (80, 33),
            (80, 34),
            (80, 35),
            (80, 36),
            (81, 0),
            (81, 1),
            (81, 2),
            (81, 3),
            (81, 4),
            (81, 5),
            (81, 6),
            (81, 7),
            (81, 8),
            (81, 9),
            (81, 10),
            (81, 11),
            (81, 12),
            (81, 13),
            (81, 14),
            (81, 15),
            (81, 16),
            (81, 17),
            (81, 18),
            (81, 19),
            (81, 20),
            (81, 21),
            (81, 22),
            (81, 23),
            (81, 24),
            (81, 25),
            (81, 26),
            (81, 27),
            (81, 28),
            (81, 29),
            (81, 30),
            (81, 31),
            (81, 32),
            (81, 33),
            (81, 34),
            (81, 35),
            (81, 36),
            (82, 0),
            (82, 1),
            (82, 2),
            (82, 3),
            (82, 4),
            (82, 5),
            (82, 6),
            (82, 7),
            (82, 8),
            (82, 9),
            (82, 10),
            (82, 11),
            (82, 12),
            (82, 13),
            (82, 14),
            (82, 15),
            (82, 16),
            (82, 17),
            (82, 18),
            (82, 19),
            (82, 20),
            (82, 21),
            (82, 22),
            (82, 23),
            (82, 24),
            (82, 25),
            (82, 26),
            (82, 27),
            (82, 28),
            (82, 29),
            (82, 30),
            (82, 31),
            (82, 32),
            (82, 33),
            (82, 34),
            (82, 35),
            (82, 36),
            (82, 37),
            (83, 0),
            (83, 1),
            (83, 2),
            (83, 3),
            (83, 4),
            (83, 5),
            (83, 6),
            (83, 7),
            (83, 8),
            (83, 9),
            (83, 10),
            (83, 11),
            (83, 12),
            (83, 13),
            (83, 14),
            (83, 15),
            (83, 16),
            (83, 17),
            (83, 18),
            (83, 19),
            (83, 20),
            (83, 21),
            (83, 22),
            (83, 23),
            (83, 24),
            (83, 25),
            (83, 26),
            (83, 27),
            (83, 28),
            (83, 29),
            (83, 30),
            (83, 31),
            (83, 32),
            (83, 33),
            (83, 34),
            (83, 35),
            (83, 36),
            (83, 37),
            (84, 0),
            (84, 1),
            (84, 2),
            (84, 3),
            (84, 4),
            (84, 5),
            (84, 6),
            (84, 7),
            (84, 8),
            (84, 9),
            (84, 10),
            (84, 11),
            (84, 12),
            (84, 13),
            (84, 14),
            (84, 15),
            (84, 16),
            (84, 17),
            (84, 18),
            (84, 19),
            (84, 20),
            (84, 21),
            (84, 22),
            (84, 23),
            (84, 24),
            (84, 25),
            (84, 26),
            (84, 27),
            (84, 28),
            (84, 29),
            (84, 30),
            (84, 31),
            (84, 32),
            (84, 33),
            (84, 34),
            (84, 35),
            (84, 36),
            (84, 37),
            (85, 0),
            (85, 1),
            (85, 2),
            (85, 3),
            (85, 4),
            (85, 5),
            (85, 6),
            (85, 7),
            (85, 8),
            (85, 9),
            (85, 10),
            (85, 11),
            (85, 12),
            (85, 13),
            (85, 14),
            (85, 15),
            (85, 16),
            (85, 17),
            (85, 18),
            (85, 19),
            (85, 20),
            (85, 21),
            (85, 22),
            (85, 23),
            (85, 24),
            (85, 25),
            (85, 26),
            (85, 27),
            (85, 28),
            (85, 29),
            (85, 30),
            (85, 31),
            (85, 32),
            (85, 33),
            (85, 34),
            (85, 35),
            (85, 36),
            (85, 37),
            (85, 38),
            (86, 0),
            (86, 1),
            (86, 2),
            (86, 3),
            (86, 4),
            (86, 5),
            (86, 6),
            (86, 7),
            (86, 8),
            (86, 9),
            (86, 10),
            (86, 11),
            (86, 12),
            (86, 13),
            (86, 14),
            (86, 15),
            (86, 16),
            (86, 17),
            (86, 18),
            (86, 19),
            (86, 20),
            (86, 21),
            (86, 22),
            (86, 23),
            (86, 24),
            (86, 25),
            (86, 26),
            (86, 27),
            (86, 28),
            (86, 29),
            (86, 30),
            (86, 31),
            (86, 32),
            (86, 33),
            (86, 34),
            (86, 35),
            (86, 36),
            (86, 37),
            (86, 38),
            (87, 0),
            (87, 1),
            (87, 2),
            (87, 3),
            (87, 4),
            (87, 5),
            (87, 6),
            (87, 7),
            (87, 8),
            (87, 9),
            (87, 10),
            (87, 11),
            (87, 12),
            (87, 13),
            (87, 14),
            (87, 15),
            (87, 16),
            (87, 17),
            (87, 18),
            (87, 19),
            (87, 20),
            (87, 21),
            (87, 22),
            (87, 23),
            (87, 24),
            (87, 25),
            (87, 26),
            (87, 27),
            (87, 28),
            (87, 29),
            (87, 30),
            (87, 31),
            (87, 32),
            (87, 33),
            (87, 34),
            (87, 35),
            (87, 36),
            (87, 37),
            (87, 38),
            (87, 39),
            (88, 0),
            (88, 1),
            (88, 2),
            (88, 3),
            (88, 4),
            (88, 5),
            (88, 6),
            (88, 7),
            (88, 8),
            (88, 9),
            (88, 10),
            (88, 11),
            (88, 12),
            (88, 13),
            (88, 14),
            (88, 15),
            (88, 16),
            (88, 17),
            (88, 18),
            (88, 19),
            (88, 20),
            (88, 21),
            (88, 22),
            (88, 23),
            (88, 24),
            (88, 25),
            (88, 26),
            (88, 27),
            (88, 28),
            (88, 29),
            (88, 30),
            (88, 31),
            (88, 32),
            (88, 33),
            (88, 34),
            (88, 35),
            (88, 36),
            (88, 37),
            (88, 38),
            (88, 39),
            (89, 0),
            (89, 1),
            (89, 2),
            (89, 3),
            (89, 4),
            (89, 5),
            (89, 6),
            (89, 7),
            (89, 8),
            (89, 9),
            (89, 10),
            (89, 11),
            (89, 12),
            (89, 13),
            (89, 14),
            (89, 15),
            (89, 16),
            (89, 17),
            (89, 18),
            (89, 19),
            (89, 20),
            (89, 21),
            (89, 22),
            (89, 23),
            (89, 24),
            (89, 25),
            (89, 26),
            (89, 27),
            (89, 28),
            (89, 29),
            (89, 30),
            (89, 31),
            (89, 32),
            (89, 33),
            (89, 34),
            (89, 35),
            (89, 36),
            (89, 37),
            (89, 38),
            (89, 39),
            (89, 40),
            (90, 0),
            (90, 1),
            (90, 2),
            (90, 3),
            (90, 4),
            (90, 5),
            (90, 6),
            (90, 7),
            (90, 8),
            (90, 9),
            (90, 10),
            (90, 11),
            (90, 12),
            (90, 13),
            (90, 14),
            (90, 15),
            (90, 16),
            (90, 17),
            (90, 18),
            (90, 19),
            (90, 20),
            (90, 21),
            (90, 22),
            (90, 23),
            (90, 24),
            (90, 25),
            (90, 26),
            (90, 27),
            (90, 28),
            (90, 29),
            (90, 30),
            (90, 31),
            (90, 32),
            (90, 33),
            (90, 34),
            (90, 35),
            (90, 36),
            (90, 37),
            (90, 38),
            (90, 39),
            (90, 40),
            (91, 0),
            (91, 1),
            (91, 2),
            (91, 3),
            (91, 4),
            (91, 5),
            (91, 6),
            (91, 7),
            (91, 8),
            (91, 9),
            (91, 10),
            (91, 11),
            (91, 12),
            (91, 13),
            (91, 14),
            (91, 15),
            (91, 16),
            (91, 17),
            (91, 18),
            (91, 19),
            (91, 20),
            (91, 21),
            (91, 22),
            (91, 23),
            (91, 24),
            (91, 25),
            (91, 26),
            (91, 27),
            (91, 28),
            (91, 29),
            (91, 30),
            (91, 31),
            (91, 32),
            (91, 33),
            (91, 34),
            (91, 35),
            (91, 36),
            (91, 37),
            (91, 38),
            (91, 39),
            (91, 40),
            (91, 41),
            (92, 0),
            (92, 1),
            (92, 2),
            (92, 3),
            (92, 4),
            (92, 5),
            (92, 6),
            (92, 7),
            (92, 8),
            (92, 9),
            (92, 10),
            (92, 11),
            (92, 12),
            (92, 13),
            (92, 14),
            (92, 15),
            (92, 16),
            (92, 17),
            (92, 18),
            (92, 19),
            (92, 20),
            (92, 21),
            (92, 22),
            (92, 23),
            (92, 24),
            (92, 25),
            (92, 26),
            (92, 27),
            (92, 28),
            (92, 29),
            (92, 30),
            (92, 31),
            (92, 32),
            (92, 33),
            (92, 34),
            (92, 35),
            (92, 36),
            (92, 37),
            (92, 38),
            (92, 39),
            (92, 40),
            (92, 41),
            (93, 0),
            (93, 1),
            (93, 2),
            (93, 3),
            (93, 4),
            (93, 5),
            (93, 6),
            (93, 7),
            (93, 8),
            (93, 9),
            (93, 10),
            (93, 11),
            (93, 12),
            (93, 13),
            (93, 14),
            (93, 15),
            (93, 16),
            (93, 17),
            (93, 18),
            (93, 19),
            (93, 20),
            (93, 21),
            (93, 22),
            (93, 23),
            (93, 24),
            (93, 25),
            (93, 26),
            (93, 27),
            (93, 28),
            (93, 29),
            (93, 30),
            (93, 31),
            (93, 32),
            (93, 33),
            (93, 34),
            (93, 35),
            (93, 36),
            (93, 37),
            (93, 38),
            (93, 39),
            (93, 40),
            (93, 41),
            (94, 0),
            (94, 1),
            (94, 2),
            (94, 3),
            (94, 4),
            (94, 5),
            (94, 6),
            (94, 7),
            (94, 8),
            (94, 9),
            (94, 10),
            (94, 11),
            (94, 12),
            (94, 13),
            (94, 14),
            (94, 15),
            (94, 16),
            (94, 17),
            (94, 18),
            (94, 19),
            (94, 20),
            (94, 21),
            (94, 22),
            (94, 23),
            (94, 24),
            (94, 25),
            (94, 26),
            (94, 27),
            (94, 28),
            (94, 29),
            (94, 30),
            (94, 31),
            (94, 32),
            (94, 33),
            (94, 34),
            (94, 35),
            (94, 36),
            (94, 37),
            (94, 38),
            (94, 39),
            (94, 40),
            (94, 41),
            (94, 42),
            (95, 0),
            (95, 1),
            (95, 2),
            (95, 3),
            (95, 4),
            (95, 5),
            (95, 6),
            (95, 7),
            (95, 8),
            (95, 9),
            (95, 10),
            (95, 11),
            (95, 12),
            (95, 13),
            (95, 14),
            (95, 15),
            (95, 16),
            (95, 17),
            (95, 18),
            (95, 19),
            (95, 20),
            (95, 21),
            (95, 22),
            (95, 23),
            (95, 24),
            (95, 25),
            (95, 26),
            (95, 27),
            (95, 28),
            (95, 29),
            (95, 30),
            (95, 31),
            (95, 32),
            (95, 33),
            (95, 34),
            (95, 35),
            (95, 36),
            (95, 37),
            (95, 38),
            (95, 39),
            (95, 40),
            (95, 41),
            (95, 42),
            (96, 0),
            (96, 1),
            (96, 2),
            (96, 3),
            (96, 4),
            (96, 5),
            (96, 6),
            (96, 7),
            (96, 8),
            (96, 9),
            (96, 10),
            (96, 11),
            (96, 12),
            (96, 13),
            (96, 14),
            (96, 15),
            (96, 16),
            (96, 17),
            (96, 18),
            (96, 19),
            (96, 20),
            (96, 21),
            (96, 22),
            (96, 23),
            (96, 24),
            (96, 25),
            (96, 26),
            (96, 27),
            (96, 28),
            (96, 29),
            (96, 30),
            (96, 31),
            (96, 32),
            (96, 33),
            (96, 34),
            (96, 35),
            (96, 36),
            (96, 37),
            (96, 38),
            (96, 39),
            (96, 40),
            (96, 41),
            (96, 42),
            (96, 43),
            (97, 0),
            (97, 1),
            (97, 2),
            (97, 3),
            (97, 4),
            (97, 5),
            (97, 6),
            (97, 7),
            (97, 8),
            (97, 9),
            (97, 10),
            (97, 11),
            (97, 12),
            (97, 13),
            (97, 14),
            (97, 15),
            (97, 16),
            (97, 17),
            (97, 18),
            (97, 19),
            (97, 20),
            (97, 21),
            (97, 22),
            (97, 23),
            (97, 24),
            (97, 25),
            (97, 26),
            (97, 27),
            (97, 28),
            (97, 29),
            (97, 30),
            (97, 31),
            (97, 32),
            (97, 33),
            (97, 34),
            (97, 35),
            (97, 36),
            (97, 37),
            (97, 38),
            (97, 39),
            (97, 40),
            (97, 41),
            (97, 42),
            (97, 43),
            (98, 0),
            (98, 1),
            (98, 2),
            (98, 3),
            (98, 4),
            (98, 5),
            (98, 6),
            (98, 7),
            (98, 8),
            (98, 9),
            (98, 10),
            (98, 11),
            (98, 12),
            (98, 13),
            (98, 14),
            (98, 15),
            (98, 16),
            (98, 17),
            (98, 18),
            (98, 19),
            (98, 20),
            (98, 21),
            (98, 22),
            (98, 23),
            (98, 24),
            (98, 25),
            (98, 26),
            (98, 27),
            (98, 28),
            (98, 29),
            (98, 30),
            (98, 31),
            (98, 32),
            (98, 33),
            (98, 34),
            (98, 35),
            (98, 36),
            (98, 37),
            (98, 38),
            (98, 39),
            (98, 40),
            (98, 41),
            (98, 42),
            (98, 43),
            (98, 44),
            (99, 0),
            (99, 1),
            (99, 2),
            (99, 3),
            (99, 4),
            (99, 5),
            (99, 6),
            (99, 7),
            (99, 8),
            (99, 9),
            (99, 10),
            (99, 11),
            (99, 12),
            (99, 13),
            (99, 14),
            (99, 15),
            (99, 16),
            (99, 17),
            (99, 18),
            (99, 19),
            (99, 20),
            (99, 21),
            (99, 22),
            (99, 23),
            (99, 24),
            (99, 25),
            (99, 26),
            (99, 27),
            (99, 28),
            (99, 29),
            (99, 30),
            (99, 31),
            (99, 32),
            (99, 33),
            (99, 34),
            (99, 35),
            (99, 36),
            (99, 37),
            (99, 38),
            (99, 39),
            (99, 40),
            (99, 41),
            (99, 42),
            (99, 43),
            (99, 44),
            (100, 0),
            (100, 1),
            (100, 2),
            (100, 3),
            (100, 4),
            (100, 5),
            (100, 6),
            (100, 7),
            (100, 8),
            (100, 9),
            (100, 10),
            (100, 11),
            (100, 12),
            (100, 13),
            (100, 14),
            (100, 15),
            (100, 16),
            (100, 17),
            (100, 18),
            (100, 19),
            (100, 20),
            (100, 21),
            (100, 22),
            (100, 23),
            (100, 24),
            (100, 25),
            (100, 26),
            (100, 27),
            (100, 28),
            (100, 29),
            (100, 30),
            (100, 31),
            (100, 32),
            (100, 33),
            (100, 34),
            (100, 35),
            (100, 36),
            (100, 37),
            (100, 38),
            (100, 39),
            (100, 40),
            (100, 41),
            (100, 42),
            (100, 43),
            (100, 44),
            (100, 45),
            (101, 0),
            (101, 1),
            (101, 2),
            (101, 3),
            (101, 4),
            (101, 5),
            (101, 6),
            (101, 7),
            (101, 8),
            (101, 9),
            (101, 10),
            (101, 11),
            (101, 12),
            (101, 13),
            (101, 14),
            (101, 15),
            (101, 16),
            (101, 17),
            (101, 18),
            (101, 19),
            (101, 20),
            (101, 21),
            (101, 22),
            (101, 23),
            (101, 24),
            (101, 25),
            (101, 26),
            (101, 27),
            (101, 28),
            (101, 29),
            (101, 30),
            (101, 31),
            (101, 32),
            (101, 33),
            (101, 34),
            (101, 35),
            (101, 36),
            (101, 37),
            (101, 38),
            (101, 39),
            (101, 40),
            (101, 41),
            (101, 42),
            (101, 43),
            (101, 44),
            (101, 45),
            (102, 0),
            (102, 1),
            (102, 2),
            (102, 3),
            (102, 4),
            (102, 5),
            (102, 6),
            (102, 7),
            (102, 8),
            (102, 9),
            (102, 10),
            (102, 11),
            (102, 12),
            (102, 13),
            (102, 14),
            (102, 15),
            (102, 16),
            (102, 17),
            (102, 18),
            (102, 19),
            (102, 20),
            (102, 21),
            (102, 22),
            (102, 23),
            (102, 24),
            (102, 25),
            (102, 26),
            (102, 27),
            (102, 28),
            (102, 29),
            (102, 30),
            (102, 31),
            (102, 32),
            (102, 33),
            (102, 34),
            (102, 35),
            (102, 36),
            (102, 37),
            (102, 38),
            (102, 39),
            (102, 40),
            (102, 41),
            (102, 42),
            (102, 43),
            (102, 44),
            (102, 45),
            (102, 46),
            (103, 0),
            (103, 1),
            (103, 2),
            (103, 3),
            (103, 4),
            (103, 5),
            (103, 6),
            (103, 7),
            (103, 8),
            (103, 9),
            (103, 10),
            (103, 11),
            (103, 12),
            (103, 13),
            (103, 14),
            (103, 15),
            (103, 16),
            (103, 17),
            (103, 18),
            (103, 19),
            (103, 20),
            (103, 21),
            (103, 22),
            (103, 23),
            (103, 24),
            (103, 25),
            (103, 26),
            (103, 27),
            (103, 28),
            (103, 29),
            (103, 30),
            (103, 31),
            (103, 32),
            (103, 33),
            (103, 34),
            (103, 35),
            (103, 36),
            (103, 37),
            (103, 38),
            (103, 39),
            (103, 40),
            (103, 41),
            (103, 42),
            (103, 43),
            (103, 44),
            (103, 45),
            (103, 46),
            (104, 0),
            (104, 1),
            (104, 2),
            (104, 3),
            (104, 4),
            (104, 5),
            (104, 6),
            (104, 7),
            (104, 8),
            (104, 9),
            (104, 10),
            (104, 11),
            (104, 12),
            (104, 13),
            (104, 14),
            (104, 15),
            (104, 16),
            (104, 17),
            (104, 18),
            (104, 19),
            (104, 20),
            (104, 21),
            (104, 22),
            (104, 23),
            (104, 24),
            (104, 25),
            (104, 26),
            (104, 27),
            (104, 28),
            (104, 29),
            (104, 30),
            (104, 31),
            (104, 32),
            (104, 33),
            (104, 34),
            (104, 35),
            (104, 36),
            (104, 37),
            (104, 38),
            (104, 39),
            (104, 40),
            (104, 41),
            (104, 42),
            (104, 43),
            (104, 44),
            (104, 45),
            (104, 46),
            (105, 0),
            (105, 1),
            (105, 2),
            (105, 3),
            (105, 4),
            (105, 5),
            (105, 6),
            (105, 7),
            (105, 8),
            (105, 9),
            (105, 10),
            (105, 11),
            (105, 12),
            (105, 13),
            (105, 14),
            (105, 15),
            (105, 16),
            (105, 17),
            (105, 18),
            (105, 19),
            (105, 20),
            (105, 21),
            (105, 22),
            (105, 23),
            (105, 24),
            (105, 25),
            (105, 26),
            (105, 27),
            (105, 28),
            (105, 29),
            (105, 30),
            (105, 31),
            (105, 32),
            (105, 33),
            (105, 34),
            (105, 35),
            (105, 36),
            (105, 37),
            (105, 38),
            (105, 39),
            (105, 40),
            (105, 41),
            (105, 42),
            (105, 43),
            (105, 44),
            (105, 45),
            (105, 46),
            (105, 47),
            (106, 0),
            (106, 1),
            (106, 2),
            (106, 3),
            (106, 4),
            (106, 5),
            (106, 6),
            (106, 7),
            (106, 8),
            (106, 9),
            (106, 10),
            (106, 11),
            (106, 12),
            (106, 13),
            (106, 14),
            (106, 15),
            (106, 16),
            (106, 17),
            (106, 18),
            (106, 19),
            (106, 20),
            (106, 21),
            (106, 22),
            (106, 23),
            (106, 24),
            (106, 25),
            (106, 26),
            (106, 27),
            (106, 28),
            (106, 29),
            (106, 30),
            (106, 31),
            (106, 32),
            (106, 33),
            (106, 34),
            (106, 35),
            (106, 36),
            (106, 37),
            (106, 38),
            (106, 39),
            (106, 40),
            (106, 41),
            (106, 42),
            (106, 43),
            (106, 44),
            (106, 45),
            (106, 46),
            (106, 47),
            (107, 0),
            (107, 1),
            (107, 2),
            (107, 3),
            (107, 4),
            (107, 5),
            (107, 6),
            (107, 7),
            (107, 8),
            (107, 9),
            (107, 10),
            (107, 11),
            (107, 12),
            (107, 13),
            (107, 14),
            (107, 15),
            (107, 16),
            (107, 17),
            (107, 18),
            (107, 19),
            (107, 20),
            (107, 21),
            (107, 22),
            (107, 23),
            (107, 24),
            (107, 25),
            (107, 26),
            (107, 27),
            (107, 28),
            (107, 29),
            (107, 30),
            (107, 31),
            (107, 32),
            (107, 33),
            (107, 34),
            (107, 35),
            (107, 36),
            (107, 37),
            (107, 38),
            (107, 39),
            (107, 40),
            (107, 41),
            (107, 42),
            (107, 43),
            (107, 44),
            (107, 45),
            (107, 46),
            (107, 47),
            (107, 48),
            (108, 0),
            (108, 1),
            (108, 2),
            (108, 3),
            (108, 4),
            (108, 5),
            (108, 6),
            (108, 7),
            (108, 8),
            (108, 9),
            (108, 10),
            (108, 11),
            (108, 12),
            (108, 13),
            (108, 14),
            (108, 15),
            (108, 16),
            (108, 17),
            (108, 18),
            (108, 19),
            (108, 20),
            (108, 21),
            (108, 22),
            (108, 23),
            (108, 24),
            (108, 25),
            (108, 26),
            (108, 27),
            (108, 28),
            (108, 29),
            (108, 30),
            (108, 31),
            (108, 32),
            (108, 33),
            (108, 34),
            (108, 35),
            (108, 36),
            (108, 37),
            (108, 38),
            (108, 39),
            (108, 40),
            (108, 41),
            (108, 42),
            (108, 43),
            (108, 44),
            (108, 45),
            (108, 46),
            (108, 47),
            (108, 48),
            (109, 0),
            (109, 1),
            (109, 2),
            (109, 3),
            (109, 4),
            (109, 5),
            (109, 6),
            (109, 7),
            (109, 8),
            (109, 9),
            (109, 10),
            (109, 11),
            (109, 12),
            (109, 13),
            (109, 14),
            (109, 15),
            (109, 16),
            (109, 17),
            (109, 18),
            (109, 19),
            (109, 20),
            (109, 21),
            (109, 22),
            (109, 23),
            (109, 24),
            (109, 25),
            (109, 26),
            (109, 27),
            (109, 28),
            (109, 29),
            (109, 30),
            (109, 31),
            (109, 32),
            (109, 33),
            (109, 34),
            (109, 35),
            (109, 36),
            (109, 37),
            (109, 38),
            (109, 39),
            (109, 40),
            (109, 41),
            (109, 42),
            (109, 43),
            (109, 44),
            (109, 45),
            (109, 46),
            (109, 47),
            (109, 48),
            (109, 49),
            (110, 0),
            (110, 1),
            (110, 2),
            (110, 3),
            (110, 4),
            (110, 5),
            (110, 6),
            (110, 7),
            (110, 8),
            (110, 9),
            (110, 10),
            (110, 11),
            (110, 12),
            (110, 13),
            (110, 14),
            (110, 15),
            (110, 16),
            (110, 17),
            (110, 18),
            (110, 19),
            (110, 20),
            (110, 21),
            (110, 22),
            (110, 23),
            (110, 24),
            (110, 25),
            (110, 26),
            (110, 27),
            (110, 28),
            (110, 29),
            (110, 30),
            (110, 31),
            (110, 32),
            (110, 33),
            (110, 34),
            (110, 35),
            (110, 36),
            (110, 37),
            (110, 38),
            (110, 39),
            (110, 40),
            (110, 41),
            (110, 42),
            (110, 43),
            (110, 44),
            (110, 45),
            (110, 46),
            (110, 47),
            (110, 48),
            (110, 49),
            (111, 0),
            (111, 1),
            (111, 2),
            (111, 3),
            (111, 4),
            (111, 5),
            (111, 6),
            (111, 7),
            (111, 8),
            (111, 9),
            (111, 10),
            (111, 11),
            (111, 12),
            (111, 13),
            (111, 14),
            (111, 15),
            (111, 16),
            (111, 17),
            (111, 18),
            (111, 19),
            (111, 20),
            (111, 21),
            (111, 22),
            (111, 23),
            (111, 24),
            (111, 25),
            (111, 26),
            (111, 27),
            (111, 28),
            (111, 29),
            (111, 30),
            (111, 31),
            (111, 32),
            (111, 33),
            (111, 34),
            (111, 35),
            (111, 36),
            (111, 37),
            (111, 38),
            (111, 39),
            (111, 40),
            (111, 41),
            (111, 42),
            (111, 43),
            (111, 44),
            (111, 45),
            (111, 46),
            (111, 47),
            (111, 48),
            (111, 49),
            (111, 50),
            (112, 0),
            (112, 1),
            (112, 2),
            (112, 3),
            (112, 4),
            (112, 5),
            (112, 6),
            (112, 7),
            (112, 8),
            (112, 9),
            (112, 10),
            (112, 11),
            (112, 12),
            (112, 13),
            (112, 14),
            (112, 15),
            (112, 16),
            (112, 17),
            (112, 18),
            (112, 19),
            (112, 20),
            (112, 21),
            (112, 22),
            (112, 23),
            (112, 24),
            (112, 25),
            (112, 26),
            (112, 27),
            (112, 28),
            (112, 29),
            (112, 30),
            (112, 31),
            (112, 32),
            (112, 33),
            (112, 34),
            (112, 35),
            (112, 36),
            (112, 37),
            (112, 38),
            (112, 39),
            (112, 40),
            (112, 41),
            (112, 42),
            (112, 43),
            (112, 44),
            (112, 45),
            (112, 46),
            (112, 47),
            (112, 48),
            (112, 49),
            (112, 50),
            (113, 0),
            (113, 1),
            (113, 2),
            (113, 3),
            (113, 4),
            (113, 5),
            (113, 6),
            (113, 7),
            (113, 8),
            (113, 9),
            (113, 10),
            (113, 11),
            (113, 12),
            (113, 13),
            (113, 14),
            (113, 15),
            (113, 16),
            (113, 17),
            (113, 18),
            (113, 19),
            (113, 20),
            (113, 21),
            (113, 22),
            (113, 23),
            (113, 24),
            (113, 25),
            (113, 26),
            (113, 27),
            (113, 28),
            (113, 29),
            (113, 30),
            (113, 31),
            (113, 32),
            (113, 33),
            (113, 34),
            (113, 35),
            (113, 36),
            (113, 37),
            (113, 38),
            (113, 39),
            (113, 40),
            (113, 41),
            (113, 42),
            (113, 43),
            (113, 44),
            (113, 45),
            (113, 46),
            (113, 47),
            (113, 48),
            (113, 49),
            (113, 50),
            (114, 0),
            (114, 1),
            (114, 2),
            (114, 3),
            (114, 4),
            (114, 5),
            (114, 6),
            (114, 7),
            (114, 8),
            (114, 9),
            (114, 10),
            (114, 11),
            (114, 12),
            (114, 13),
            (114, 14),
            (114, 15),
            (114, 16),
            (114, 17),
            (114, 18),
            (114, 19),
            (114, 20),
            (114, 21),
            (114, 22),
            (114, 23),
            (114, 24),
            (114, 25),
            (114, 26),
            (114, 27),
            (114, 28),
            (114, 29),
            (114, 30),
            (114, 31),
            (114, 32),
            (114, 33),
            (114, 34),
            (114, 35),
            (114, 36),
            (114, 37),
            (114, 38),
            (114, 39),
            (114, 40),
            (114, 41),
            (114, 42),
            (114, 43),
            (114, 44),
            (114, 45),
            (114, 46),
            (114, 47),
            (114, 48),
            (114, 49),
            (114, 50),
            (114, 51),
            (115, 0),
            (115, 1),
            (115, 2),
            (115, 3),
            (115, 4),
            (115, 5),
            (115, 6),
            (115, 7),
            (115, 8),
            (115, 9),
            (115, 10),
            (115, 11),
            (115, 12),
            (115, 13),
            (115, 14),
            (115, 15),
            (115, 16),
            (115, 17),
            (115, 18),
            (115, 19),
            (115, 20),
            (115, 21),
            (115, 22),
            (115, 23),
            (115, 24),
            (115, 25),
            (115, 26),
            (115, 27),
            (115, 28),
            (115, 29),
            (115, 30),
            (115, 31),
            (115, 32),
            (115, 33),
            (115, 34),
            (115, 35),
            (115, 36),
            (115, 37),
            (115, 38),
            (115, 39),
            (115, 40),
            (115, 41),
            (115, 42),
            (115, 43),
            (115, 44),
            (115, 45),
            (115, 46),
            (115, 47),
            (115, 48),
            (115, 49),
            (115, 50),
            (115, 51),
            (116, 0),
            (116, 1),
            (116, 2),
            (116, 3),
            (116, 4),
            (116, 5),
            (116, 6),
            (116, 7),
            (116, 8),
            (116, 9),
            (116, 10),
            (116, 11),
            (116, 12),
            (116, 13),
            (116, 14),
            (116, 15),
            (116, 16),
            (116, 17),
            (116, 18),
            (116, 19),
            (116, 20),
            (116, 21),
            (116, 22),
            (116, 23),
            (116, 24),
            (116, 25),
            (116, 26),
            (116, 27),
            (116, 28),
            (116, 29),
            (116, 30),
            (116, 31),
            (116, 32),
            (116, 33),
            (116, 34),
            (116, 35),
            (116, 36),
            (116, 37),
            (116, 38),
            (116, 39),
            (116, 40),
            (116, 41),
            (116, 42),
            (116, 43),
            (116, 44),
            (116, 45),
            (116, 46),
            (116, 47),
            (116, 48),
            (116, 49),
            (116, 50),
            (116, 51),
            (116, 52),
            (117, 0),
            (117, 1),
            (117, 2),
            (117, 3),
            (117, 4),
            (117, 5),
            (117, 6),
            (117, 7),
            (117, 8),
            (117, 9),
            (117, 10),
            (117, 11),
            (117, 12),
            (117, 13),
            (117, 14),
            (117, 15),
            (117, 16),
            (117, 17),
            (117, 18),
            (117, 19),
            (117, 20),
            (117, 21),
            (117, 22),
            (117, 23),
            (117, 24),
            (117, 25),
            (117, 26),
            (117, 27),
            (117, 28),
            (117, 29),
            (117, 30),
            (117, 31),
            (117, 32),
            (117, 33),
            (117, 34),
            (117, 35),
            (117, 36),
            (117, 37),
            (117, 38),
            (117, 39),
            (117, 40),
            (117, 41),
            (117, 42),
            (117, 43),
            (117, 44),
            (117, 45),
            (117, 46),
            (117, 47),
            (117, 48),
            (117, 49),
            (117, 50),
            (117, 51),
            (117, 52),
            (118, 0),
            (118, 1),
            (118, 2),
            (118, 3),
            (118, 4),
            (118, 5),
            (118, 6),
            (118, 7),
            (118, 8),
            (118, 9),
            (118, 10),
            (118, 11),
            (118, 12),
            (118, 13),
            (118, 14),
            (118, 15),
            (118, 16),
            (118, 17),
            (118, 18),
            (118, 19),
            (118, 20),
            (118, 21),
            (118, 22),
            (118, 23),
            (118, 24),
            (118, 25),
            (118, 26),
            (118, 27),
            (118, 28),
            (118, 29),
            (118, 30),
            (118, 31),
            (118, 32),
            (118, 33),
            (118, 34),
            (118, 35),
            (118, 36),
            (118, 37),
            (118, 38),
            (118, 39),
            (118, 40),
            (118, 41),
            (118, 42),
            (118, 43),
            (118, 44),
            (118, 45),
            (118, 46),
            (118, 47),
            (118, 48),
            (118, 49),
            (118, 50),
            (118, 51),
            (118, 52),
            (118, 53),
            (119, 0),
            (119, 1),
            (119, 2),
            (119, 3),
            (119, 4),
            (119, 5),
            (119, 6),
            (119, 7),
            (119, 8),
            (119, 9),
            (119, 10),
            (119, 11),
            (119, 12),
            (119, 13),
            (119, 14),
            (119, 15),
            (119, 16),
            (119, 17),
            (119, 18),
            (119, 19),
            (119, 20),
            (119, 21),
            (119, 22),
            (119, 23),
            (119, 24),
            (119, 25),
            (119, 26),
            (119, 27),
            (119, 28),
            (119, 29),
            (119, 30),
            (119, 31),
            (119, 32),
            (119, 33),
            (119, 34),
            (119, 35),
            (119, 36),
            (119, 37),
            (119, 38),
            (119, 39),
            (119, 40),
            (119, 41),
            (119, 42),
            (119, 43),
            (119, 44),
            (119, 45),
            (119, 46),
            (119, 47),
            (119, 48),
            (119, 49),
            (119, 50),
            (119, 51),
            (119, 52),
            (119, 53),
            (120, 0),
            (120, 1),
            (120, 2),
            (120, 3),
            (120, 4),
            (120, 5),
            (120, 6),
            (120, 7),
            (120, 8),
            (120, 9),
            (120, 10),
            (120, 11),
            (120, 12),
            (120, 13),
            (120, 14),
            (120, 15),
            (120, 16),
            (120, 17),
            (120, 18),
            (120, 19),
            (120, 20),
            (120, 21),
            (120, 22),
            (120, 23),
            (120, 24),
            (120, 25),
            (120, 26),
            (120, 27),
            (120, 28),
            (120, 29),
            (120, 30),
            (120, 31),
            (120, 32),
            (120, 33),
            (120, 34),
            (120, 35),
            (120, 36),
            (120, 37),
            (120, 38),
            (120, 39),
            (120, 40),
            (120, 41),
            (120, 42),
            (120, 43),
            (120, 44),
            (120, 45),
            (120, 46),
            (120, 47),
            (120, 48),
            (120, 49),
            (120, 50),
            (120, 51),
            (120, 52),
            (120, 53),
            (120, 54),
            (121, 0),
            (121, 1),
            (121, 2),
            (121, 3),
            (121, 4),
            (121, 5),
            (121, 6),
            (121, 7),
            (121, 8),
            (121, 9),
            (121, 10),
            (121, 11),
            (121, 12),
            (121, 13),
            (121, 14),
            (121, 15),
            (121, 16),
            (121, 17),
            (121, 18),
            (121, 19),
            (121, 20),
            (121, 21),
            (121, 22),
            (121, 23),
            (121, 24),
            (121, 25),
            (121, 26),
            (121, 27),
            (121, 28),
            (121, 29),
            (121, 30),
            (121, 31),
            (121, 32),
            (121, 33),
            (121, 34),
            (121, 35),
            (121, 36),
            (121, 37),
            (121, 38),
            (121, 39),
            (121, 40),
            (121, 41),
            (121, 42),
            (121, 43),
            (121, 44),
            (121, 45),
            (121, 46),
            (121, 47),
            (121, 48),
            (121, 49),
            (121, 50),
            (121, 51),
            (121, 52),
            (121, 53),
            (121, 54),
            (122, 0),
            (122, 1),
            (122, 2),
            (122, 3),
            (122, 4),
            (122, 5),
            (122, 6),
            (122, 7),
            (122, 8),
            (122, 9),
            (122, 10),
            (122, 11),
            (122, 12),
            (122, 13),
            (122, 14),
            (122, 15),
            (122, 16),
            (122, 17),
            (122, 18),
            (122, 19),
            (122, 20),
            (122, 21),
            (122, 22),
            (122, 23),
            (122, 24),
            (122, 25),
            (122, 26),
            (122, 27),
            (122, 28),
            (122, 29),
            (122, 30),
            (122, 31),
            (122, 32),
            (122, 33),
            (122, 34),
            (122, 35),
            (122, 36),
            (122, 37),
            (122, 38),
            (122, 39),
            (122, 40),
            (122, 41),
            (122, 42),
            (122, 43),
            (122, 44),
            (122, 45),
            (122, 46),
            (122, 47),
            (122, 48),
            (122, 49),
            (122, 50),
            (122, 51),
            (122, 52),
            (122, 53),
            (122, 54),
            (122, 55),
            (123, 0),
            (123, 1),
            (123, 2),
            (123, 3),
            (123, 4),
            (123, 5),
            (123, 6),
            (123, 7),
            (123, 8),
            (123, 9),
            (123, 10),
            (123, 11),
            (123, 12),
            (123, 13),
            (123, 14),
            (123, 15),
            (123, 16),
            (123, 17),
            (123, 18),
            (123, 19),
            (123, 20),
            (123, 21),
            (123, 22),
            (123, 23),
            (123, 24),
            (123, 25),
            (123, 26),
            (123, 27),
            (123, 28),
            (123, 29),
            (123, 30),
            (123, 31),
            (123, 32),
            (123, 33),
            (123, 34),
            (123, 35),
            (123, 36),
            (123, 37),
            (123, 38),
            (123, 39),
            (123, 40),
            (123, 41),
            (123, 42),
            (123, 43),
            (123, 44),
            (123, 45),
            (123, 46),
            (123, 47),
            (123, 48),
            (123, 49),
            (123, 50),
            (123, 51),
            (123, 52),
            (123, 53),
            (123, 54),
            (123, 55),
            (124, 0),
            (124, 1),
            (124, 2),
            (124, 3),
            (124, 4),
            (124, 5),
            (124, 6),
            (124, 7),
            (124, 8),
            (124, 9),
            (124, 10),
            (124, 11),
            (124, 12),
            (124, 13),
            (124, 14),
            (124, 15),
            (124, 16),
            (124, 17),
            (124, 18),
            (124, 19),
            (124, 20),
            (124, 21),
            (124, 22),
            (124, 23),
            (124, 24),
            (124, 25),
            (124, 26),
            (124, 27),
            (124, 28),
            (124, 29),
            (124, 30),
            (124, 31),
            (124, 32),
            (124, 33),
            (124, 34),
            (124, 35),
            (124, 36),
            (124, 37),
            (124, 38),
            (124, 39),
            (124, 40),
            (124, 41),
            (124, 42),
            (124, 43),
            (124, 44),
            (124, 45),
            (124, 46),
            (124, 47),
            (124, 48),
            (124, 49),
            (124, 50),
            (124, 51),
            (124, 52),
            (124, 53),
            (124, 54),
            (124, 55),
            (125, 0),
            (125, 1),
            (125, 2),
            (125, 3),
            (125, 4),
            (125, 5),
            (125, 6),
            (125, 7),
            (125, 8),
            (125, 9),
            (125, 10),
            (125, 11),
            (125, 12),
            (125, 13),
            (125, 14),
            (125, 15),
            (125, 16),
            (125, 17),
            (125, 18),
            (125, 19),
            (125, 20),
            (125, 21),
            (125, 22),
            (125, 23),
            (125, 24),
            (125, 25),
            (125, 26),
            (125, 27),
            (125, 28),
            (125, 29),
            (125, 30),
            (125, 31),
            (125, 32),
            (125, 33),
            (125, 34),
            (125, 35),
            (125, 36),
            (125, 37),
            (125, 38),
            (125, 39),
            (125, 40),
            (125, 41),
            (125, 42),
            (125, 43),
            (125, 44),
            (125, 45),
            (125, 46),
            (125, 47),
            (125, 48),
            (125, 49),
            (125, 50),
            (125, 51),
            (125, 52),
            (125, 53),
            (125, 54),
            (125, 55),
            (125, 56),
            (126, 0),
            (126, 1),
            (126, 2),
            (126, 3),
            (126, 4),
            (126, 5),
            (126, 6),
            (126, 7),
            (126, 8),
            (126, 9),
            (126, 10),
            (126, 11),
            (126, 12),
            (126, 13),
            (126, 14),
            (126, 15),
            (126, 16),
            (126, 17),
            (126, 18),
            (126, 19),
            (126, 20),
            (126, 21),
            (126, 22),
            (126, 23),
            (126, 24),
            (126, 25),
            (126, 26),
            (126, 27),
            (126, 28),
            (126, 29),
            (126, 30),
            (126, 31),
            (126, 32),
            (126, 33),
            (126, 34),
            (126, 35),
            (126, 36),
            (126, 37),
            (126, 38),
            (126, 39),
            (126, 40),
            (126, 41),
            (126, 42),
            (126, 43),
            (126, 44),
            (126, 45),
            (126, 46),
            (126, 47),
            (126, 48),
            (126, 49),
            (126, 50),
            (126, 51),
            (126, 52),
            (126, 53),
            (126, 54),
            (126, 55),
            (126, 56),
            (127, 0),
            (127, 1),
            (127, 2),
            (127, 3),
            (127, 4),
            (127, 5),
            (127, 6),
            (127, 7),
            (127, 8),
            (127, 9),
            (127, 10),
            (127, 11),
            (127, 12),
            (127, 13),
            (127, 14),
            (127, 15),
            (127, 16),
            (127, 17),
            (127, 18),
            (127, 19),
            (127, 20),
            (127, 21),
            (127, 22),
            (127, 23),
            (127, 24),
            (127, 25),
            (127, 26),
            (127, 27),
            (127, 28),
            (127, 29),
            (127, 30),
            (127, 31),
            (127, 32),
            (127, 33),
            (127, 34),
            (127, 35),
            (127, 36),
            (127, 37),
            (127, 38),
            (127, 39),
            (127, 40),
            (127, 41),
            (127, 42),
            (127, 43),
            (127, 44),
            (127, 45),
            (127, 46),
            (127, 47),
            (127, 48),
            (127, 49),
            (127, 50),
            (127, 51),
            (127, 52),
            (127, 53),
            (127, 54),
            (127, 55),
            (127, 56),
            (127, 57),
            (128, 0),
            (128, 1),
            (128, 2),
            (128, 3),
            (128, 4),
            (128, 5),
            (128, 6),
            (128, 7),
            (128, 8),
            (128, 9),
            (128, 10),
            (128, 11),
            (128, 12),
            (128, 13),
            (128, 14),
            (128, 15),
            (128, 16),
            (128, 17),
            (128, 18),
            (128, 19),
            (128, 20),
            (128, 21),
            (128, 22),
            (128, 23),
            (128, 24),
            (128, 25),
            (128, 26),
            (128, 27),
            (128, 28),
            (128, 29),
            (128, 30),
            (128, 31),
            (128, 32),
            (128, 33),
            (128, 34),
            (128, 35),
            (128, 36),
            (128, 37),
            (128, 38),
            (128, 39),
            (128, 40),
            (128, 41),
            (128, 42),
            (128, 43),
            (128, 44),
            (128, 45),
            (128, 46),
            (128, 47),
            (128, 48),
            (128, 49),
            (128, 50),
            (128, 51),
            (128, 52),
            (128, 53),
            (128, 54),
            (128, 55),
            (128, 56),
            (128, 57),
            (129, 0),
            (129, 1),
            (129, 2),
            (129, 3),
            (129, 4),
            (129, 5),
            (129, 6),
            (129, 7),
            (129, 8),
            (129, 9),
            (129, 10),
            (129, 11),
            (129, 12),
            (129, 13),
            (129, 14),
            (129, 15),
            (129, 16),
            (129, 17),
            (129, 18),
            (129, 19),
            (129, 20),
            (129, 21),
            (129, 22),
            (129, 23),
            (129, 24),
            (129, 25),
            (129, 26),
            (129, 27),
            (129, 28),
            (129, 29),
            (129, 30),
            (129, 31),
            (129, 32),
            (129, 33),
            (129, 34),
            (129, 35),
            (129, 36),
            (129, 37),
            (129, 38),
            (129, 39),
            (129, 40),
            (129, 41),
            (129, 42),
            (129, 43),
            (129, 44),
            (129, 45),
            (129, 46),
            (129, 47),
            (129, 48),
            (129, 49),
            (129, 50),
            (129, 51),
            (129, 52),
            (129, 53),
            (129, 54),
            (129, 55),
            (129, 56),
            (129, 57),
            (129, 58),
            (130, 0),
            (130, 1),
            (130, 2),
            (130, 3),
            (130, 4),
            (130, 5),
            (130, 6),
            (130, 7),
            (130, 8),
            (130, 9),
            (130, 10),
            (130, 11),
            (130, 12),
            (130, 13),
            (130, 14),
            (130, 15),
            (130, 16),
            (130, 17),
            (130, 18),
            (130, 19),
            (130, 20),
            (130, 21),
            (130, 22),
            (130, 23),
            (130, 24),
            (130, 25),
            (130, 26),
            (130, 27),
            (130, 28),
            (130, 29),
            (130, 30),
            (130, 31),
            (130, 32),
            (130, 33),
            (130, 34),
            (130, 35),
            (130, 36),
            (130, 37),
            (130, 38),
            (130, 39),
            (130, 40),
            (130, 41),
            (130, 42),
            (130, 43),
            (130, 44),
            (130, 45),
            (130, 46),
            (130, 47),
            (130, 48),
            (130, 49),
            (130, 50),
            (130, 51),
            (130, 52),
            (130, 53),
            (130, 54),
            (130, 55),
            (130, 56),
            (130, 57),
            (130, 58),
            (131, 0),
            (131, 1),
            (131, 2),
            (131, 3),
            (131, 4),
            (131, 5),
            (131, 6),
            (131, 7),
            (131, 8),
            (131, 9),
            (131, 10),
            (131, 11),
            (131, 12),
            (131, 13),
            (131, 14),
            (131, 15),
            (131, 16),
            (131, 17),
            (131, 18),
            (131, 19),
            (131, 20),
            (131, 21),
            (131, 22),
            (131, 23),
            (131, 24),
            (131, 25),
            (131, 26),
            (131, 27),
            (131, 28),
            (131, 29),
            (131, 30),
            (131, 31),
            (131, 32),
            (131, 33),
            (131, 34),
            (131, 35),
            (131, 36),
            (131, 37),
            (131, 38),
            (131, 39),
            (131, 40),
            (131, 41),
            (131, 42),
            (131, 43),
            (131, 44),
            (131, 45),
            (131, 46),
            (131, 47),
            (131, 48),
            (131, 49),
            (131, 50),
            (131, 51),
            (131, 52),
            (131, 53),
            (131, 54),
            (131, 55),
            (131, 56),
            (131, 57),
            (131, 58),
            (131, 59),
            (132, 0),
            (132, 1),
            (132, 2),
            (132, 3),
            (132, 4),
            (132, 5),
            (132, 6),
            (132, 7),
            (132, 8),
            (132, 9),
            (132, 10),
            (132, 11),
            (132, 12),
            (132, 13),
            (132, 14),
            (132, 15),
            (132, 16),
            (132, 17),
            (132, 18),
            (132, 19),
            (132, 20),
            (132, 21),
            (132, 22),
            (132, 23),
            (132, 24),
            (132, 25),
            (132, 26),
            (132, 27),
            (132, 28),
            (132, 29),
            (132, 30),
            (132, 31),
            (132, 32),
            (132, 33),
            (132, 34),
            (132, 35),
            (132, 36),
            (132, 37),
            (132, 38),
            (132, 39),
            (132, 40),
            (132, 41),
            (132, 42),
            (132, 43),
            (132, 44),
            (132, 45),
            (132, 46),
            (132, 47),
            (132, 48),
            (132, 49),
            (132, 50),
            (132, 51),
            (132, 52),
            (132, 53),
            (132, 54),
            (132, 55),
            (132, 56),
            (132, 57),
            (132, 58),
            (132, 59),
            (133, 0),
            (133, 1),
            (133, 2),
            (133, 3),
            (133, 4),
            (133, 5),
            (133, 6),
            (133, 7),
            (133, 8),
            (133, 9),
            (133, 10),
            (133, 11),
            (133, 12),
            (133, 13),
            (133, 14),
            (133, 15),
            (133, 16),
            (133, 17),
            (133, 18),
            (133, 19),
            (133, 20),
            (133, 21),
            (133, 22),
            (133, 23),
            (133, 24),
            (133, 25),
            (133, 26),
            (133, 27),
            (133, 28),
            (133, 29),
            (133, 30),
            (133, 31),
            (133, 32),
            (133, 33),
            (133, 34),
            (133, 35),
            (133, 36),
            (133, 37),
            (133, 38),
            (133, 39),
            (133, 40),
            (133, 41),
            (133, 42),
            (133, 43),
            (133, 44),
            (133, 45),
            (133, 46),
            (133, 47),
            (133, 48),
            (133, 49),
            (133, 50),
            (133, 51),
            (133, 52),
            (133, 53),
            (133, 54),
            (133, 55),
            (133, 56),
            (133, 57),
            (133, 58),
            (133, 59),
            (134, 0),
            (134, 1),
            (134, 2),
            (134, 3),
            (134, 4),
            (134, 5),
            (134, 6),
            (134, 7),
            (134, 8),
            (134, 9),
            (134, 10),
            (134, 11),
            (134, 12),
            (134, 13),
            (134, 14),
            (134, 15),
            (134, 16),
            (134, 17),
            (134, 18),
            (134, 19),
            (134, 20),
            (134, 21),
            (134, 22),
            (134, 23),
            (134, 24),
            (134, 25),
            (134, 26),
            (134, 27),
            (134, 28),
            (134, 29),
            (134, 30),
            (134, 31),
            (134, 32),
            (134, 33),
            (134, 34),
            (134, 35),
            (134, 36),
            (134, 37),
            (134, 38),
            (134, 39),
            (134, 40),
            (134, 41),
            (134, 42),
            (134, 43),
            (134, 44),
            (134, 45),
            (134, 46),
            (134, 47),
            (134, 48),
            (134, 49),
            (134, 50),
            (134, 51),
            (134, 52),
            (134, 53),
            (134, 54),
            (134, 55),
            (134, 56),
            (134, 57),
            (134, 58),
            (134, 59),
            (134, 60),
            (135, 0),
            (135, 1),
            (135, 2),
            (135, 3),
            (135, 4),
            (135, 5),
            (135, 6),
            (135, 7),
            (135, 8),
            (135, 9),
            (135, 10),
            (135, 11),
            (135, 12),
            (135, 13),
            (135, 14),
            (135, 15),
            (135, 16),
            (135, 17),
            (135, 18),
            (135, 19),
            (135, 20),
            (135, 21),
            (135, 22),
            (135, 23),
            (135, 24),
            (135, 25),
            (135, 26),
            (135, 27),
            (135, 28),
            (135, 29),
            (135, 30),
            (135, 31),
            (135, 32),
            (135, 33),
            (135, 34),
            (135, 35),
            (135, 36),
            (135, 37),
            (135, 38),
            (135, 39),
            (135, 40),
            (135, 41),
            (135, 42),
            (135, 43),
            (135, 44),
            (135, 45),
            (135, 46),
            (135, 47),
            (135, 48),
            (135, 49),
            (135, 50),
            (135, 51),
            (135, 52),
            (135, 53),
            (135, 54),
            (135, 55),
            (135, 56),
            (135, 57),
            (135, 58),
            (135, 59),
            (135, 60),
            (136, 0),
            (136, 1),
            (136, 2),
            (136, 3),
            (136, 4),
            (136, 5),
            (136, 6),
            (136, 7),
            (136, 8),
            (136, 9),
            (136, 10),
            (136, 11),
            (136, 12),
            (136, 13),
            (136, 14),
            (136, 15),
            (136, 16),
            (136, 17),
            (136, 18),
            (136, 19),
            (136, 20),
            (136, 21),
            (136, 22),
            (136, 23),
            (136, 24),
            (136, 25),
            (136, 26),
            (136, 27),
            (136, 28),
            (136, 29),
            (136, 30),
            (136, 31),
            (136, 32),
            (136, 33),
            (136, 34),
            (136, 35),
            (136, 36),
            (136, 37),
            (136, 38),
            (136, 39),
            (136, 40),
            (136, 41),
            (136, 42),
            (136, 43),
            (136, 44),
            (136, 45),
            (136, 46),
            (136, 47),
            (136, 48),
            (136, 49),
            (136, 50),
            (136, 51),
            (136, 52),
            (136, 53),
            (136, 54),
            (136, 55),
            (136, 56),
            (136, 57),
            (136, 58),
            (136, 59),
            (136, 60),
            (136, 61),
            (137, 0),
            (137, 1),
            (137, 2),
            (137, 3),
            (137, 4),
            (137, 5),
            (137, 6),
            (137, 7),
            (137, 8),
            (137, 9),
            (137, 10),
            (137, 11),
            (137, 12),
            (137, 13),
            (137, 14),
            (137, 15),
            (137, 16),
            (137, 17),
            (137, 18),
            (137, 19),
            (137, 20),
            (137, 21),
            (137, 22),
            (137, 23),
            (137, 24),
            (137, 25),
            (137, 26),
            (137, 27),
            (137, 28),
            (137, 29),
            (137, 30),
            (137, 31),
            (137, 32),
            (137, 33),
            (137, 34),
            (137, 35),
            (137, 36),
            (137, 37),
            (137, 38),
            (137, 39),
            (137, 40),
            (137, 41),
            (137, 42),
            (137, 43),
            (137, 44),
            (137, 45),
            (137, 46),
            (137, 47),
            (137, 48),
            (137, 49),
            (137, 50),
            (137, 51),
            (137, 52),
            (137, 53),
            (137, 54),
            (137, 55),
            (137, 56),
            (137, 57),
            (137, 58),
            (137, 59),
            (137, 60),
            (137, 61),
            (138, 0),
            (138, 1),
            (138, 2),
            (138, 3),
            (138, 4),
            (138, 5),
            (138, 6),
            (138, 7),
            (138, 8),
            (138, 9),
            (138, 10),
            (138, 11),
            (138, 12),
            (138, 13),
            (138, 14),
            (138, 15),
            (138, 16),
            (138, 17),
            (138, 18),
            (138, 19),
            (138, 20),
            (138, 21),
            (138, 22),
            (138, 23),
            (138, 24),
            (138, 25),
            (138, 26),
            (138, 27),
            (138, 28),
            (138, 29),
            (138, 30),
            (138, 31),
            (138, 32),
            (138, 33),
            (138, 34),
            (138, 35),
            (138, 36),
            (138, 37),
            (138, 38),
            (138, 39),
            (138, 40),
            (138, 41),
            (138, 42),
            (138, 43),
            (138, 44),
            (138, 45),
            (138, 46),
            (138, 47),
            (138, 48),
            (138, 49),
            (138, 50),
            (138, 51),
            (138, 52),
            (138, 53),
            (138, 54),
            (138, 55),
            (138, 56),
            (138, 57),
            (138, 58),
            (138, 59),
            (138, 60),
            (138, 61),
            (138, 62),
            (139, 0),
            (139, 1),
            (139, 2),
            (139, 3),
            (139, 4),
            (139, 5),
            (139, 6),
            (139, 7),
            (139, 8),
            (139, 9),
            (139, 10),
            (139, 11),
            (139, 12),
            (139, 13),
            (139, 14),
            (139, 15),
            (139, 16),
            (139, 17),
            (139, 18),
            (139, 19),
            (139, 20),
            (139, 21),
            (139, 22),
            (139, 23),
            (139, 24),
            (139, 25),
            (139, 26),
            (139, 27),
            (139, 28),
            (139, 29),
            (139, 30),
            (139, 31),
            (139, 32),
            (139, 33),
            (139, 34),
            (139, 35),
            (139, 36),
            (139, 37),
            (139, 38),
            (139, 39),
            (139, 40),
            (139, 41),
            (139, 42),
            (139, 43),
            (139, 44),
            (139, 45),
            (139, 46),
            (139, 47),
            (139, 48),
            (139, 49),
            (139, 50),
            (139, 51),
            (139, 52),
            (139, 53),
            (139, 54),
            (139, 55),
            (139, 56),
            (139, 57),
            (139, 58),
            (139, 59),
            (139, 60),
            (139, 61),
            (139, 62),
            (140, 0),
            (140, 1),
            (140, 2),
            (140, 3),
            (140, 4),
            (140, 5),
            (140, 6),
            (140, 7),
            (140, 8),
            (140, 9),
            (140, 10),
            (140, 11),
            (140, 12),
            (140, 13),
            (140, 14),
            (140, 15),
            (140, 16),
            (140, 17),
            (140, 18),
            (140, 19),
            (140, 20),
            (140, 21),
            (140, 22),
            (140, 23),
            (140, 24),
            (140, 25),
            (140, 26),
            (140, 27),
            (140, 28),
            (140, 29),
            (140, 30),
            (140, 31),
            (140, 32),
            (140, 33),
            (140, 34),
            (140, 35),
            (140, 36),
            (140, 37),
            (140, 38),
            (140, 39),
            (140, 40),
            (140, 41),
            (140, 42),
            (140, 43),
            (140, 44),
            (140, 45),
            (140, 46),
            (140, 47),
            (140, 48),
            (140, 49),
            (140, 50),
            (140, 51),
            (140, 52),
            (140, 53),
            (140, 54),
            (140, 55),
            (140, 56),
            (140, 57),
            (140, 58),
            (140, 59),
            (140, 60),
            (140, 61),
            (140, 62),
            (140, 63),
            (141, 0),
            (141, 1),
            (141, 2),
            (141, 3),
            (141, 4),
            (141, 5),
            (141, 6),
            (141, 7),
            (141, 8),
            (141, 9),
            (141, 10),
            (141, 11),
            (141, 12),
            (141, 13),
            (141, 14),
            (141, 15),
            (141, 16),
            (141, 17),
            (141, 18),
            (141, 19),
            (141, 20),
            (141, 21),
            (141, 22),
            (141, 23),
            (141, 24),
            (141, 25),
            (141, 26),
            (141, 27),
            (141, 28),
            (141, 29),
            (141, 30),
            (141, 31),
            (141, 32),
            (141, 33),
            (141, 34),
            (141, 35),
            (141, 36),
            (141, 37),
            (141, 38),
            (141, 39),
            (141, 40),
            (141, 41),
            (141, 42),
            (141, 43),
            (141, 44),
            (141, 45),
            (141, 46),
            (141, 47),
            (141, 48),
            (141, 49),
            (141, 50),
            (141, 51),
            (141, 52),
            (141, 53),
            (141, 54),
            (141, 55),
            (141, 56),
            (141, 57),
            (141, 58),
            (141, 59),
            (141, 60),
            (141, 61),
            (141, 62),
            (141, 63),
            (142, 0),
            (142, 1),
            (142, 2),
            (142, 3),
            (142, 4),
            (142, 5),
            (142, 6),
            (142, 7),
            (142, 8),
            (142, 9),
            (142, 10),
            (142, 11),
            (142, 12),
            (142, 13),
            (142, 14),
            (142, 15),
            (142, 16),
            (142, 17),
            (142, 18),
            (142, 19),
            (142, 20),
            (142, 21),
            (142, 22),
            (142, 23),
            (142, 24),
            (142, 25),
            (142, 26),
            (142, 27),
            (142, 28),
            (142, 29),
            (142, 30),
            (142, 31),
            (142, 32),
            (142, 33),
            (142, 34),
            (142, 35),
            (142, 36),
            (142, 37),
            (142, 38),
            (142, 39),
            (142, 40),
            (142, 41),
            (142, 42),
            (142, 43),
            (142, 44),
            (142, 45),
            (142, 46),
            (142, 47),
            (142, 48),
            (142, 49),
            (142, 50),
            (142, 51),
            (142, 52),
            (142, 53),
            (142, 54),
            (142, 55),
            (142, 56),
            (142, 57),
            (142, 58),
            (142, 59),
            (142, 60),
            (142, 61),
            (142, 62),
            (142, 63),
            (143, 0),
            (143, 1),
            (143, 2),
            (143, 3),
            (143, 4),
            (143, 5),
            (143, 6),
            (143, 7),
            (143, 8),
            (143, 9),
            (143, 10),
            (143, 11),
            (143, 12),
            (143, 13),
            (143, 14),
            (143, 15),
            (143, 16),
            (143, 17),
            (143, 18),
            (143, 19),
            (143, 20),
            (143, 21),
            (143, 22),
            (143, 23),
            (143, 24),
            (143, 25),
            (143, 26),
            (143, 27),
            (143, 28),
            (143, 29),
            (143, 30),
            (143, 31),
            (143, 32),
            (143, 33),
            (143, 34),
            (143, 35),
            (143, 36),
            (143, 37),
            (143, 38),
            (143, 39),
            (143, 40),
            (143, 41),
            (143, 42),
            (143, 43),
            (143, 44),
            (143, 45),
            (143, 46),
            (143, 47),
            (143, 48),
            (143, 49),
            (143, 50),
            (143, 51),
            (143, 52),
            (143, 53),
            (143, 54),
            (143, 55),
            (143, 56),
            (143, 57),
            (143, 58),
            (143, 59),
            (143, 60),
            (143, 61),
            (143, 62),
            (143, 63),
            (143, 64),
            (144, 0),
            (144, 1),
            (144, 2),
            (144, 3),
            (144, 4),
            (144, 5),
            (144, 6),
            (144, 7),
            (144, 8),
            (144, 9),
            (144, 10),
            (144, 11),
            (144, 12),
            (144, 13),
            (144, 14),
            (144, 15),
            (144, 16),
            (144, 17),
            (144, 18),
            (144, 19),
            (144, 20),
            (144, 21),
            (144, 22),
            (144, 23),
            (144, 24),
            (144, 25),
            (144, 26),
            (144, 27),
            (144, 28),
            (144, 29),
            (144, 30),
            (144, 31),
            (144, 32),
            (144, 33),
            (144, 34),
            (144, 35),
            (144, 36),
            (144, 37),
            (144, 38),
            (144, 39),
            (144, 40),
            (144, 41),
            (144, 42),
            (144, 43),
            (144, 44),
            (144, 45),
            (144, 46),
            (144, 47),
            (144, 48),
            (144, 49),
            (144, 50),
            (144, 51),
            (144, 52),
            (144, 53),
            (144, 54),
            (144, 55),
            (144, 56),
            (144, 57),
            (144, 58),
            (144, 59),
            (144, 60),
            (144, 61),
            (144, 62),
            (144, 63),
            (144, 64),
            (145, 0),
            (145, 1),
            (145, 2),
            (145, 3),
            (145, 4),
            (145, 5),
            (145, 6),
            (145, 7),
            (145, 8),
            (145, 9),
            (145, 10),
            (145, 11),
            (145, 12),
            (145, 13),
            (145, 14),
            (145, 15),
            (145, 16),
            (145, 17),
            (145, 18),
            (145, 19),
            (145, 20),
            (145, 21),
            (145, 22),
            (145, 23),
            (145, 24),
            (145, 25),
            (145, 26),
            (145, 27),
            (145, 28),
            (145, 29),
            (145, 30),
            (145, 31),
            (145, 32),
            (145, 33),
            (145, 34),
            (145, 35),
            (145, 36),
            (145, 37),
            (145, 38),
            (145, 39),
            (145, 40),
            (145, 41),
            (145, 42),
            (145, 43),
            (145, 44),
            (145, 45),
            (145, 46),
            (145, 47),
            (145, 48),
            (145, 49),
            (145, 50),
            (145, 51),
            (145, 52),
            (145, 53),
            (145, 54),
            (145, 55),
            (145, 56),
            (145, 57),
            (145, 58),
            (145, 59),
            (145, 60),
            (145, 61),
            (145, 62),
            (145, 63),
            (145, 64),
            (145, 65),
            (146, 0),
            (146, 1),
            (146, 2),
            (146, 3),
            (146, 4),
            (146, 5),
            (146, 6),
            (146, 7),
            (146, 8),
            (146, 9),
            (146, 10),
            (146, 11),
            (146, 12),
            (146, 13),
            (146, 14),
            (146, 15),
            (146, 16),
            (146, 17),
            (146, 18),
            (146, 19),
            (146, 20),
            (146, 21),
            (146, 22),
            (146, 23),
            (146, 24),
            (146, 25),
            (146, 26),
            (146, 27),
            (146, 28),
            (146, 29),
            (146, 30),
            (146, 31),
            (146, 32),
            (146, 33),
            (146, 34),
            (146, 35),
            (146, 36),
            (146, 37),
            (146, 38),
            (146, 39),
            (146, 40),
            (146, 41),
            (146, 42),
            (146, 43),
            (146, 44),
            (146, 45),
            (146, 46),
            (146, 47),
            (146, 48),
            (146, 49),
            (146, 50),
            (146, 51),
            (146, 52),
            (146, 53),
            (146, 54),
            (146, 55),
            (146, 56),
            (146, 57),
            (146, 58),
            (146, 59),
            (146, 60),
            (146, 61),
            (146, 62),
            (146, 63),
            (146, 64),
            (146, 65),
            (147, 0),
            (147, 1),
            (147, 2),
            (147, 3),
            (147, 4),
            (147, 5),
            (147, 6),
            (147, 7),
            (147, 8),
            (147, 9),
            (147, 10),
            (147, 11),
            (147, 12),
            (147, 13),
            (147, 14),
            (147, 15),
            (147, 16),
            (147, 17),
            (147, 18),
            (147, 19),
            (147, 20),
            (147, 21),
            (147, 22),
            (147, 23),
            (147, 24),
            (147, 25),
            (147, 26),
            (147, 27),
            (147, 28),
            (147, 29),
            (147, 30),
            (147, 31),
            (147, 32),
            (147, 33),
            (147, 34),
            (147, 35),
            (147, 36),
            (147, 37),
            (147, 38),
            (147, 39),
            (147, 40),
            (147, 41),
            (147, 42),
            (147, 43),
            (147, 44),
            (147, 45),
            (147, 46),
            (147, 47),
            (147, 48),
            (147, 49),
            (147, 50),
            (147, 51),
            (147, 52),
            (147, 53),
            (147, 54),
            (147, 55),
            (147, 56),
            (147, 57),
            (147, 58),
            (147, 59),
            (147, 60),
            (147, 61),
            (147, 62),
            (147, 63),
            (147, 64),
            (147, 65),
            (147, 66),
            (148, 0),
            (148, 1),
            (148, 2),
            (148, 3),
            (148, 4),
            (148, 5),
            (148, 6),
            (148, 7),
            (148, 8),
            (148, 9),
            (148, 10),
            (148, 11),
            (148, 12),
            (148, 13),
            (148, 14),
            (148, 15),
            (148, 16),
            (148, 17),
            (148, 18),
            (148, 19),
            (148, 20),
            (148, 21),
            (148, 22),
            (148, 23),
            (148, 24),
            (148, 25),
            (148, 26),
            (148, 27),
            (148, 28),
            (148, 29),
            (148, 30),
            (148, 31),
            (148, 32),
            (148, 33),
            (148, 34),
            (148, 35),
            (148, 36),
            (148, 37),
            (148, 38),
            (148, 39),
            (148, 40),
            (148, 41),
            (148, 42),
            (148, 43),
            (148, 44),
            (148, 45),
            (148, 46),
            (148, 47),
            (148, 48),
            (148, 49),
            (148, 50),
            (148, 51),
            (148, 52),
            (148, 53),
            (148, 54),
            (148, 55),
            (148, 56),
            (148, 57),
            (148, 58),
            (148, 59),
            (148, 60),
            (148, 61),
            (148, 62),
            (148, 63),
            (148, 64),
            (148, 65),
            (148, 66),
            (149, 0),
            (149, 1),
            (149, 2),
            (149, 3),
            (149, 4),
            (149, 5),
            (149, 6),
            (149, 7),
            (149, 8),
            (149, 9),
            (149, 10),
            (149, 11),
            (149, 12),
            (149, 13),
            (149, 14),
            (149, 15),
            (149, 16),
            (149, 17),
            (149, 18),
            (149, 19),
            (149, 20),
            (149, 21),
            (149, 22),
            (149, 23),
            (149, 24),
            (149, 25),
            (149, 26),
            (149, 27),
            (149, 28),
            (149, 29),
            (149, 30),
            (149, 31),
            (149, 32),
            (149, 33),
            (149, 34),
            (149, 35),
            (149, 36),
            (149, 37),
            (149, 38),
            (149, 39),
            (149, 40),
            (149, 41),
            (149, 42),
            (149, 43),
            (149, 44),
            (149, 45),
            (149, 46),
            (149, 47),
            (149, 48),
            (149, 49),
            (149, 50),
            (149, 51),
            (149, 52),
            (149, 53),
            (149, 54),
            (149, 55),
            (149, 56),
            (149, 57),
            (149, 58),
            (149, 59),
            (149, 60),
            (149, 61),
            (149, 62),
            (149, 63),
            (149, 64),
            (149, 65),
            (149, 66),
            (149, 67),
        }:
            return 38
        elif key in {
            (4, 2),
            (6, 3),
            (13, 6),
            (15, 7),
            (24, 11),
            (26, 12),
            (33, 15),
            (35, 16),
            (44, 20),
            (46, 21),
            (53, 24),
            (55, 25),
            (64, 29),
            (66, 30),
            (73, 33),
            (75, 34),
            (84, 38),
            (86, 39),
            (93, 42),
            (95, 43),
            (104, 47),
            (106, 48),
            (113, 51),
            (115, 52),
            (124, 56),
            (133, 60),
            (135, 61),
            (142, 64),
            (144, 65),
        }:
            return 48
        return 11

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_7_output, num_attn_2_2_output):
        key = (num_attn_1_7_output, num_attn_2_2_output)
        return 0

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_6_output, num_attn_1_4_output):
        key = (num_attn_2_6_output, num_attn_1_4_output)
        return 11

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_6_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_4_output, num_attn_2_1_output):
        key = (num_attn_2_4_output, num_attn_2_1_output)
        return 26

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_4_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_3_output_scores = classifier_weights.loc[
        [("num_mlp_2_3_outputs", str(v)) for v in num_mlp_2_3_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                num_mlp_0_2_output_scores,
                num_mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                num_mlp_1_2_output_scores,
                num_mlp_1_3_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                mlp_2_2_output_scores,
                mlp_2_3_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                num_mlp_2_2_output_scores,
                num_mlp_2_3_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "3", "</s>"]))