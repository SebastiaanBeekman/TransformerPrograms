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
        "output/length/rasp/sort/trainlength30/s1/sort_weights.csv",
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
        if q_position in {0}:
            return k_position == 2
        elif q_position in {8, 1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {35, 3}:
            return k_position == 6
        elif q_position in {33, 4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {19, 37, 6}:
            return k_position == 8
        elif q_position in {38, 7}:
            return k_position == 11
        elif q_position in {9, 31}:
            return k_position == 5
        elif q_position in {10, 15}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 3
        elif q_position in {16, 25, 12}:
            return k_position == 15
        elif q_position in {13, 14}:
            return k_position == 12
        elif q_position in {17, 20}:
            return k_position == 16
        elif q_position in {18, 30}:
            return k_position == 25
        elif q_position in {21}:
            return k_position == 26
        elif q_position in {22}:
            return k_position == 20
        elif q_position in {23}:
            return k_position == 28
        elif q_position in {24}:
            return k_position == 21
        elif q_position in {26, 27}:
            return k_position == 29
        elif q_position in {28}:
            return k_position == 0
        elif q_position in {29}:
            return k_position == 17
        elif q_position in {32}:
            return k_position == 36
        elif q_position in {34}:
            return k_position == 24
        elif q_position in {36, 39}:
            return k_position == 30

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 4
        elif q_position in {1, 29, 7}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {8, 34, 4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 1
        elif q_position in {11, 13, 6}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {10, 28}:
            return k_position == 13
        elif q_position in {24, 12}:
            return k_position == 19
        elif q_position in {26, 14, 22}:
            return k_position == 10
        elif q_position in {18, 15}:
            return k_position == 12
        elif q_position in {16}:
            return k_position == 20
        elif q_position in {17}:
            return k_position == 26
        elif q_position in {19, 23}:
            return k_position == 15
        elif q_position in {20}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 22
        elif q_position in {27}:
            return k_position == 16
        elif q_position in {30}:
            return k_position == 31
        elif q_position in {31}:
            return k_position == 39
        elif q_position in {32}:
            return k_position == 35
        elif q_position in {33, 35}:
            return k_position == 32
        elif q_position in {36, 37}:
            return k_position == 30
        elif q_position in {38, 39}:
            return k_position == 37

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 26}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5, 39}:
            return k_position == 8
        elif q_position in {8, 38, 6}:
            return k_position == 10
        elif q_position in {14, 7}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10, 35}:
            return k_position == 4
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {12, 30}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 18
        elif q_position in {15}:
            return k_position == 5
        elif q_position in {16}:
            return k_position == 22
        elif q_position in {17, 31, 25, 23}:
            return k_position == 13
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 21
        elif q_position in {20}:
            return k_position == 12
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {24, 22}:
            return k_position == 19
        elif q_position in {27}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 0
        elif q_position in {32, 36}:
            return k_position == 33
        elif q_position in {33}:
            return k_position == 30
        elif q_position in {34}:
            return k_position == 26
        elif q_position in {37}:
            return k_position == 32

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 28}:
            return token == "2"
        elif position in {1, 38, 17, 23, 31}:
            return token == "0"
        elif position in {24, 2, 18, 21}:
            return token == "1"
        elif position in {
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
            13,
            14,
            15,
            16,
            19,
            20,
            22,
            27,
        }:
            return token == "4"
        elif position in {25, 29}:
            return token == "<s>"
        elif position in {26}:
            return token == "</s>"
        elif position in {32, 33, 34, 35, 36, 37, 39, 30}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 16, 23, 24, 27}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 35
        elif q_position in {8, 2, 3, 7}:
            return k_position == 5
        elif q_position in {4, 13, 6}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 15
        elif q_position in {18, 11, 14, 22}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 9
        elif q_position in {19}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 28
        elif q_position in {29, 28, 21}:
            return k_position == 7
        elif q_position in {25}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 6
        elif q_position in {30}:
            return k_position == 29
        elif q_position in {31}:
            return k_position == 18
        elif q_position in {32}:
            return k_position == 37
        elif q_position in {33, 34, 37}:
            return k_position == 36
        elif q_position in {35, 36}:
            return k_position == 33
        elif q_position in {38}:
            return k_position == 30
        elif q_position in {39}:
            return k_position == 32

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
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {34, 14, 30}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 27
        elif q_position in {27}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 0
        elif q_position in {38, 31}:
            return k_position == 33
        elif q_position in {32}:
            return k_position == 30
        elif q_position in {33, 35, 36, 39}:
            return k_position == 37
        elif q_position in {37}:
            return k_position == 31

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0, 33, 32, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {1, 2, 9, 16, 18, 21}:
            return token == "1"
        elif position in {3, 4, 5, 6, 7, 8, 10, 12, 13, 15, 19, 29}:
            return token == "3"
        elif position in {27, 11, 14}:
            return token == "0"
        elif position in {17, 28}:
            return token == "2"
        elif position in {20}:
            return token == "<s>"
        elif position in {22, 23, 24, 25, 26}:
            return token == "4"

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 12}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9, 11}:
            return k_position == 13
        elif q_position in {10, 21}:
            return k_position == 12
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 16
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {16}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 20
        elif q_position in {19, 20}:
            return k_position == 22
        elif q_position in {22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 14
        elif q_position in {24, 37}:
            return k_position == 26
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {26}:
            return k_position == 28
        elif q_position in {27, 28, 39}:
            return k_position == 29
        elif q_position in {33, 36, 29, 30}:
            return k_position == 0
        elif q_position in {31}:
            return k_position == 32
        elif q_position in {32}:
            return k_position == 30
        elif q_position in {34, 35}:
            return k_position == 34
        elif q_position in {38}:
            return k_position == 25

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 36}:
            return token == "1"
        elif position in {8, 1, 9}:
            return token == "</s>"
        elif position in {2, 3, 4, 5, 6, 7, 37, 29}:
            return token == "0"
        elif position in {
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
            30,
            31,
            32,
            33,
            34,
            35,
            38,
            39,
        }:
            return token == ""

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 28}:
            return token == "1"
        elif position in {1, 2, 3, 4, 5, 34, 35, 37, 38, 31}:
            return token == ""
        elif position in {32, 36, 6, 7, 39, 30}:
            return token == "<pad>"
        elif position in {8, 9, 12, 13, 14, 16}:
            return token == "</s>"
        elif position in {10, 11, 15}:
            return token == "<s>"
        elif position in {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}:
            return token == "0"
        elif position in {29}:
            return token == "2"
        elif position in {33}:
            return token == "4"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 2, 3, 4, 5, 6, 33, 34, 35, 36, 37, 38, 39, 29, 30, 31}:
            return token == "0"
        elif position in {
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
            25,
            26,
            27,
            28,
            32,
        }:
            return token == ""

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 32, 4, 5, 6, 7, 8, 9, 36, 37, 21, 30, 31}:
            return token == ""
        elif position in {1, 2, 3, 35, 38, 39}:
            return token == "0"
        elif position in {10, 11, 12, 15}:
            return token == "</s>"
        elif position in {13}:
            return token == "<s>"
        elif position in {33, 34, 14, 16, 17, 18, 19, 20, 29}:
            return token == "3"
        elif position in {22, 23, 24, 25, 26, 28}:
            return token == "2"
        elif position in {27}:
            return token == "1"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0, 33, 34, 35, 37, 38}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {16, 2}:
            return k_position == 6
        elif q_position in {17, 3}:
            return k_position == 8
        elif q_position in {11, 4, 14}:
            return k_position == 10
        elif q_position in {36, 5, 39, 18, 31}:
            return k_position == 11
        elif q_position in {19, 6, 30}:
            return k_position == 12
        elif q_position in {8, 7}:
            return k_position == 15
        elif q_position in {9}:
            return k_position == 18
        elif q_position in {10}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 25
        elif q_position in {13}:
            return k_position == 27
        elif q_position in {15}:
            return k_position == 39
        elif q_position in {20}:
            return k_position == 7
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {22}:
            return k_position == 4
        elif q_position in {23, 25, 26, 27, 28, 29}:
            return k_position == 3
        elif q_position in {24}:
            return k_position == 1
        elif q_position in {32}:
            return k_position == 9

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0, 32, 35, 7, 30}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {8, 34, 36, 31}:
            return k_position == 12
        elif q_position in {33, 37, 38, 9, 29}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {11, 39}:
            return k_position == 15
        elif q_position in {12}:
            return k_position == 17
        elif q_position in {13}:
            return k_position == 18
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 20
        elif q_position in {16}:
            return k_position == 21
        elif q_position in {17}:
            return k_position == 23
        elif q_position in {18}:
            return k_position == 24
        elif q_position in {19}:
            return k_position == 25
        elif q_position in {20, 21}:
            return k_position == 26
        elif q_position in {22, 23}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 31
        elif q_position in {25, 26}:
            return k_position == 34
        elif q_position in {27}:
            return k_position == 37
        elif q_position in {28}:
            return k_position == 32

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 1, 2, 3, 36, 5, 39, 29, 31}:
            return token == ""
        elif position in {4}:
            return token == "<pad>"
        elif position in {26, 21, 6, 23}:
            return token == "<s>"
        elif position in {32, 34, 37, 7, 9, 10, 18, 19}:
            return token == "2"
        elif position in {33, 8, 16, 20, 22, 25}:
            return token == "1"
        elif position in {11, 12, 13, 14, 15, 17, 28}:
            return token == "0"
        elif position in {24, 27}:
            return token == "</s>"
        elif position in {38, 35, 30}:
            return token == "3"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 1, 2, 3, 32, 33, 6, 7, 34, 35, 36, 37, 38, 30}:
            return token == ""
        elif position in {39, 4, 5, 31}:
            return token == "<pad>"
        elif position in {8, 11}:
            return token == "</s>"
        elif position in {9, 10, 12}:
            return token == "<s>"
        elif position in {13, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29}:
            return token == "1"
        elif position in {20, 14, 15}:
            return token == "2"
        elif position in {16, 28}:
            return token == "0"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position, attn_0_1_output):
        key = (position, attn_0_1_output)
        if key in {
            (1, "</s>"),
            (1, "<s>"),
            (8, "2"),
            (8, "4"),
            (8, "</s>"),
            (8, "<s>"),
            (9, "2"),
            (9, "4"),
            (9, "</s>"),
            (9, "<s>"),
            (10, "2"),
            (10, "4"),
            (10, "</s>"),
            (10, "<s>"),
            (11, "2"),
            (11, "4"),
            (11, "</s>"),
            (11, "<s>"),
            (12, "2"),
            (12, "4"),
            (12, "</s>"),
            (12, "<s>"),
            (14, "2"),
            (14, "4"),
            (14, "</s>"),
            (14, "<s>"),
            (18, "2"),
            (18, "4"),
            (18, "</s>"),
            (18, "<s>"),
            (21, "2"),
            (21, "4"),
            (21, "</s>"),
            (21, "<s>"),
            (25, "2"),
            (25, "4"),
            (25, "</s>"),
            (25, "<s>"),
            (26, "2"),
            (26, "4"),
            (26, "</s>"),
            (26, "<s>"),
            (27, "2"),
            (27, "4"),
            (27, "</s>"),
            (27, "<s>"),
            (28, "2"),
            (28, "4"),
            (28, "</s>"),
        }:
            return 3
        elif key in {
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
        }:
            return 39
        elif key in {(1, "0"), (1, "1"), (1, "2"), (1, "3"), (1, "4")}:
            return 20
        return 26

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(positions, attn_0_1_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_0_output):
        key = (position, attn_0_0_output)
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
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "</s>"),
            (3, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "</s>"),
            (4, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "</s>"),
            (5, "<s>"),
            (29, "2"),
            (29, "3"),
            (29, "<s>"),
            (30, "0"),
            (30, "2"),
            (30, "3"),
            (30, "<s>"),
            (31, "0"),
            (31, "2"),
            (31, "3"),
            (31, "<s>"),
            (32, "2"),
            (32, "3"),
            (32, "<s>"),
            (33, "2"),
            (33, "3"),
            (33, "<s>"),
            (34, "0"),
            (34, "2"),
            (34, "3"),
            (34, "<s>"),
            (35, "0"),
            (35, "2"),
            (35, "3"),
            (35, "<s>"),
            (36, "0"),
            (36, "2"),
            (36, "3"),
            (36, "<s>"),
            (37, "0"),
            (37, "2"),
            (37, "3"),
            (37, "<s>"),
            (38, "0"),
            (38, "2"),
            (38, "3"),
            (38, "<s>"),
            (39, "2"),
            (39, "3"),
            (39, "<s>"),
        }:
            return 35
        elif key in {
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
            (8, "</s>"),
            (8, "<s>"),
            (30, "1"),
            (30, "4"),
            (31, "1"),
            (31, "4"),
            (33, "4"),
            (34, "1"),
            (34, "4"),
            (35, "1"),
            (35, "4"),
            (36, "1"),
            (36, "4"),
            (37, "1"),
            (37, "4"),
            (38, "1"),
            (38, "4"),
            (39, "4"),
        }:
            return 17
        return 22

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_0_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position):
        key = position
        if key in {0, 3, 4, 5, 6}:
            return 23
        elif key in {7, 29}:
            return 37
        elif key in {1, 2}:
            return 39
        return 16

    mlp_0_2_outputs = [mlp_0_2(k0) for k0 in positions]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position):
        key = position
        if key in {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}:
            return 21
        elif key in {0, 2, 3}:
            return 5
        elif key in {1}:
            return 13
        return 37

    mlp_0_3_outputs = [mlp_0_3(k0) for k0 in positions]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_4_output):
        key = (num_attn_0_5_output, num_attn_0_4_output)
        return 18

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_4_output):
        key = (num_attn_0_0_output, num_attn_0_4_output)
        return 35

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_5_output):
        key = num_attn_0_5_output
        return 7

    num_mlp_0_2_outputs = [num_mlp_0_2(k0) for k0 in num_attn_0_5_outputs]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_0_output, num_attn_0_2_output):
        key = (num_attn_0_0_output, num_attn_0_2_output)
        return 18

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 39}:
            return k_position == 1
        elif q_position in {1, 35, 9, 7}:
            return k_position == 3
        elif q_position in {2, 14}:
            return k_position == 5
        elif q_position in {32, 26, 3, 30}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {8, 5}:
            return k_position == 2
        elif q_position in {17, 6}:
            return k_position == 4
        elif q_position in {10, 12}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 9
        elif q_position in {13}:
            return k_position == 23
        elif q_position in {33, 34, 36, 38, 15}:
            return k_position == 6
        elif q_position in {16}:
            return k_position == 13
        elif q_position in {18}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 16
        elif q_position in {20}:
            return k_position == 17
        elif q_position in {21}:
            return k_position == 7
        elif q_position in {22}:
            return k_position == 20
        elif q_position in {23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 21
        elif q_position in {25}:
            return k_position == 22
        elif q_position in {27, 37}:
            return k_position == 19
        elif q_position in {28}:
            return k_position == 26
        elif q_position in {29, 31}:
            return k_position == 11

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_5_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, mlp_0_1_output):
        if position in {0, 4, 8, 9, 10, 11, 23, 30}:
            return mlp_0_1_output == 16
        elif position in {1, 3, 37, 38, 7, 16, 21, 22, 24, 25, 26, 27, 28}:
            return mlp_0_1_output == 22
        elif position in {2, 35}:
            return mlp_0_1_output == 30
        elif position in {5}:
            return mlp_0_1_output == 27
        elif position in {33, 36, 6, 13, 15, 18, 19, 20, 31}:
            return mlp_0_1_output == 17
        elif position in {12}:
            return mlp_0_1_output == 2
        elif position in {17, 34, 14, 39}:
            return mlp_0_1_output == 35
        elif position in {29}:
            return mlp_0_1_output == 20
        elif position in {32}:
            return mlp_0_1_output == 29

    attn_1_1_pattern = select_closest(mlp_0_1_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_mlp_0_3_output, k_mlp_0_3_output):
        if q_mlp_0_3_output in {0, 8, 11, 22, 23, 24}:
            return k_mlp_0_3_output == 5
        elif q_mlp_0_3_output in {1, 12, 39}:
            return k_mlp_0_3_output == 2
        elif q_mlp_0_3_output in {32, 33, 2, 34, 35, 36, 38, 27, 28, 29, 30}:
            return k_mlp_0_3_output == 3
        elif q_mlp_0_3_output in {26, 3}:
            return k_mlp_0_3_output == 16
        elif q_mlp_0_3_output in {4}:
            return k_mlp_0_3_output == 22
        elif q_mlp_0_3_output in {16, 10, 5}:
            return k_mlp_0_3_output == 37
        elif q_mlp_0_3_output in {6}:
            return k_mlp_0_3_output == 0
        elif q_mlp_0_3_output in {7}:
            return k_mlp_0_3_output == 27
        elif q_mlp_0_3_output in {9}:
            return k_mlp_0_3_output == 19
        elif q_mlp_0_3_output in {13}:
            return k_mlp_0_3_output == 13
        elif q_mlp_0_3_output in {17, 18, 14}:
            return k_mlp_0_3_output == 9
        elif q_mlp_0_3_output in {20, 15}:
            return k_mlp_0_3_output == 14
        elif q_mlp_0_3_output in {19}:
            return k_mlp_0_3_output == 23
        elif q_mlp_0_3_output in {21}:
            return k_mlp_0_3_output == 25
        elif q_mlp_0_3_output in {25}:
            return k_mlp_0_3_output == 1
        elif q_mlp_0_3_output in {31}:
            return k_mlp_0_3_output == 17
        elif q_mlp_0_3_output in {37}:
            return k_mlp_0_3_output == 4

    attn_1_2_pattern = select_closest(mlp_0_3_outputs, mlp_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_4_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_7_output, token):
        if attn_0_7_output in {"0"}:
            return token == "0"
        elif attn_0_7_output in {"1"}:
            return token == ""
        elif attn_0_7_output in {"2"}:
            return token == "3"
        elif attn_0_7_output in {"4", "3"}:
            return token == "2"
        elif attn_0_7_output in {"</s>"}:
            return token == "<s>"
        elif attn_0_7_output in {"<s>"}:
            return token == "</s>"

    attn_1_3_pattern = select_closest(tokens, attn_0_7_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_5_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, mlp_0_2_output):
        if position in {0}:
            return mlp_0_2_output == 38
        elif position in {1}:
            return mlp_0_2_output == 31
        elif position in {2, 4, 7, 8, 9, 24}:
            return mlp_0_2_output == 39
        elif position in {3}:
            return mlp_0_2_output == 27
        elif position in {5, 23}:
            return mlp_0_2_output == 18
        elif position in {6}:
            return mlp_0_2_output == 5
        elif position in {10, 21}:
            return mlp_0_2_output == 16
        elif position in {11}:
            return mlp_0_2_output == 33
        elif position in {12}:
            return mlp_0_2_output == 0
        elif position in {13}:
            return mlp_0_2_output == 23
        elif position in {14}:
            return mlp_0_2_output == 8
        elif position in {15}:
            return mlp_0_2_output == 35
        elif position in {16, 33, 32, 31}:
            return mlp_0_2_output == 2
        elif position in {17}:
            return mlp_0_2_output == 37
        elif position in {18, 20, 22}:
            return mlp_0_2_output == 34
        elif position in {19}:
            return mlp_0_2_output == 12
        elif position in {25, 28}:
            return mlp_0_2_output == 7
        elif position in {35, 26, 27, 38}:
            return mlp_0_2_output == 6
        elif position in {29}:
            return mlp_0_2_output == 4
        elif position in {36, 30}:
            return mlp_0_2_output == 3
        elif position in {34}:
            return mlp_0_2_output == 24
        elif position in {37}:
            return mlp_0_2_output == 19
        elif position in {39}:
            return mlp_0_2_output == 1

    attn_1_4_pattern = select_closest(mlp_0_2_outputs, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_1_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, attn_0_4_output):
        if position in {0, 39, 23}:
            return attn_0_4_output == "0"
        elif position in {1, 2, 3, 4, 37, 10, 11, 15, 18, 21, 22, 25, 26, 27, 28}:
            return attn_0_4_output == ""
        elif position in {31, 5, 6, 30}:
            return attn_0_4_output == "</s>"
        elif position in {7}:
            return attn_0_4_output == "4"
        elif position in {32, 35, 36, 8, 13, 16, 19}:
            return attn_0_4_output == "1"
        elif position in {9, 12, 17}:
            return attn_0_4_output == "3"
        elif position in {34, 38, 14, 20, 24, 29}:
            return attn_0_4_output == "2"
        elif position in {33}:
            return attn_0_4_output == "<s>"

    attn_1_5_pattern = select_closest(attn_0_4_outputs, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_7_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "0"
        elif q_token in {"4", "<s>", "3"}:
            return k_token == ""
        elif q_token in {"</s>"}:
            return k_token == "4"

    attn_1_6_pattern = select_closest(tokens, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_1_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, token):
        if position in {0, 32, 33, 18, 25, 29, 30}:
            return token == "2"
        elif position in {1, 19, 28, 21}:
            return token == "3"
        elif position in {2, 34, 35, 36, 37, 14, 15, 22, 27}:
            return token == "4"
        elif position in {11, 9, 3, 4}:
            return token == "0"
        elif position in {5, 38, 12, 16, 23, 24, 26, 31}:
            return token == ""
        elif position in {8, 6, 7}:
            return token == "<s>"
        elif position in {39, 10, 13, 17, 20}:
            return token == "1"

    attn_1_7_pattern = select_closest(tokens, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_5_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_6_output, position):
        if attn_0_6_output in {"1", "0", "3", "2"}:
            return position == 22
        elif attn_0_6_output in {"4"}:
            return position == 20
        elif attn_0_6_output in {"</s>"}:
            return position == 17
        elif attn_0_6_output in {"<s>"}:
            return position == 25

    num_attn_1_0_pattern = select(positions, attn_0_6_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {
            0,
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
        }:
            return token == ""
        elif position in {1}:
            return token == "<s>"
        elif position in {2}:
            return token == "3"
        elif position in {32, 33, 34, 3, 4, 5, 6, 7, 8, 9, 36, 37, 38, 39, 29, 31}:
            return token == "0"
        elif position in {10, 11}:
            return token == "</s>"
        elif position in {35, 30}:
            return token == "1"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_6_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, attn_0_3_output):
        if position in {0, 33, 2, 35, 11, 12, 15, 16, 29, 30}:
            return attn_0_3_output == "1"
        elif position in {32, 1}:
            return attn_0_3_output == "2"
        elif position in {3, 4, 37, 38, 8, 9, 10, 13, 18}:
            return attn_0_3_output == "0"
        elif position in {5, 6, 39, 21, 23, 24, 25, 26, 27}:
            return attn_0_3_output == ""
        elif position in {34, 7, 14, 20, 22, 31}:
            return attn_0_3_output == "<s>"
        elif position in {17, 19, 36}:
            return attn_0_3_output == "</s>"
        elif position in {28}:
            return attn_0_3_output == "<pad>"

    num_attn_1_2_pattern = select(attn_0_3_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_2_output, attn_0_3_output):
        if num_mlp_0_2_output in {
            0,
            2,
            4,
            5,
            6,
            7,
            8,
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
            22,
            24,
            29,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return attn_0_3_output == "2"
        elif num_mlp_0_2_output in {32, 1, 15}:
            return attn_0_3_output == "<s>"
        elif num_mlp_0_2_output in {3, 21, 23, 25, 31}:
            return attn_0_3_output == "</s>"
        elif num_mlp_0_2_output in {9}:
            return attn_0_3_output == "3"
        elif num_mlp_0_2_output in {26, 28}:
            return attn_0_3_output == "<pad>"
        elif num_mlp_0_2_output in {27}:
            return attn_0_3_output == ""
        elif num_mlp_0_2_output in {30}:
            return attn_0_3_output == "1"

    num_attn_1_3_pattern = select(
        attn_0_3_outputs, num_mlp_0_2_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, token):
        if position in {0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}:
            return token == ""
        elif position in {1, 2, 15, 16, 17, 18}:
            return token == "</s>"
        elif position in {3, 5}:
            return token == "3"
        elif position in {35, 4}:
            return token == "2"
        elif position in {32, 33, 34, 36, 6, 7, 8, 9, 38, 11, 39, 30, 31}:
            return token == "1"
        elif position in {37, 10, 12, 14, 29}:
            return token == "0"
        elif position in {13}:
            return token == "<s>"

    num_attn_1_4_pattern = select(tokens, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(num_mlp_0_2_output, num_mlp_0_1_output):
        if num_mlp_0_2_output in {0}:
            return num_mlp_0_1_output == 2
        elif num_mlp_0_2_output in {1, 33}:
            return num_mlp_0_1_output == 19
        elif num_mlp_0_2_output in {2}:
            return num_mlp_0_1_output == 8
        elif num_mlp_0_2_output in {11, 16, 3, 29}:
            return num_mlp_0_1_output == 31
        elif num_mlp_0_2_output in {4}:
            return num_mlp_0_1_output == 6
        elif num_mlp_0_2_output in {5}:
            return num_mlp_0_1_output == 38
        elif num_mlp_0_2_output in {6, 30}:
            return num_mlp_0_1_output == 3
        elif num_mlp_0_2_output in {7}:
            return num_mlp_0_1_output == 22
        elif num_mlp_0_2_output in {8, 22}:
            return num_mlp_0_1_output == 34
        elif num_mlp_0_2_output in {9}:
            return num_mlp_0_1_output == 35
        elif num_mlp_0_2_output in {10, 35}:
            return num_mlp_0_1_output == 32
        elif num_mlp_0_2_output in {12}:
            return num_mlp_0_1_output == 29
        elif num_mlp_0_2_output in {38, 13, 18, 20, 27}:
            return num_mlp_0_1_output == 7
        elif num_mlp_0_2_output in {19, 28, 14}:
            return num_mlp_0_1_output == 14
        elif num_mlp_0_2_output in {15}:
            return num_mlp_0_1_output == 12
        elif num_mlp_0_2_output in {17}:
            return num_mlp_0_1_output == 13
        elif num_mlp_0_2_output in {25, 21}:
            return num_mlp_0_1_output == 4
        elif num_mlp_0_2_output in {23}:
            return num_mlp_0_1_output == 1
        elif num_mlp_0_2_output in {24}:
            return num_mlp_0_1_output == 36
        elif num_mlp_0_2_output in {26}:
            return num_mlp_0_1_output == 18
        elif num_mlp_0_2_output in {31}:
            return num_mlp_0_1_output == 10
        elif num_mlp_0_2_output in {32, 36, 37, 39}:
            return num_mlp_0_1_output == 0
        elif num_mlp_0_2_output in {34}:
            return num_mlp_0_1_output == 20

    num_attn_1_5_pattern = select(
        num_mlp_0_1_outputs, num_mlp_0_2_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_2_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(num_mlp_0_2_output, attn_0_7_output):
        if num_mlp_0_2_output in {
            0,
            1,
            2,
            3,
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
            16,
            17,
            18,
            19,
            20,
            23,
            24,
            25,
            26,
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
        }:
            return attn_0_7_output == "0"
        elif num_mlp_0_2_output in {4, 21}:
            return attn_0_7_output == "</s>"
        elif num_mlp_0_2_output in {15, 22, 27, 28, 29}:
            return attn_0_7_output == "1"

    num_attn_1_6_pattern = select(
        attn_0_7_outputs, num_mlp_0_2_outputs, num_predicate_1_6
    )
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_2_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(mlp_0_2_output, token):
        if mlp_0_2_output in {
            0,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            18,
            23,
            24,
            25,
            27,
            28,
            29,
            30,
            33,
            34,
            35,
            37,
            38,
        }:
            return token == "1"
        elif mlp_0_2_output in {1, 36, 39, 17, 31}:
            return token == "<s>"
        elif mlp_0_2_output in {2, 26, 13, 15}:
            return token == "</s>"
        elif mlp_0_2_output in {3}:
            return token == "0"
        elif mlp_0_2_output in {4, 14, 16, 21, 22}:
            return token == ""
        elif mlp_0_2_output in {19, 20}:
            return token == "4"
        elif mlp_0_2_output in {32}:
            return token == "3"

    num_attn_1_7_pattern = select(tokens, mlp_0_2_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, ones)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(num_mlp_0_2_output):
        key = num_mlp_0_2_output
        return 37

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in num_mlp_0_2_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_0_output, attn_1_2_output):
        key = (num_mlp_0_0_output, attn_1_2_output)
        return 19

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, attn_1_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(num_mlp_0_3_output, mlp_0_3_output):
        key = (num_mlp_0_3_output, mlp_0_3_output)
        return 31

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, mlp_0_3_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_6_output, mlp_0_3_output):
        key = (attn_1_6_output, mlp_0_3_output)
        if key in {
            ("0", 2),
            ("0", 21),
            ("0", 36),
            ("1", 2),
            ("1", 21),
            ("1", 36),
            ("2", 21),
            ("3", 21),
            ("4", 2),
            ("4", 21),
            ("4", 36),
            ("</s>", 21),
            ("<s>", 21),
        }:
            return 14
        elif key in {("0", 12)}:
            return 8
        return 7

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_6_outputs, mlp_0_3_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output, num_attn_0_2_output):
        key = (num_attn_1_7_output, num_attn_0_2_output)
        return 7

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_0_2_output):
        key = (num_attn_1_2_output, num_attn_0_2_output)
        return 19

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_7_output, num_attn_1_0_output):
        key = (num_attn_1_7_output, num_attn_1_0_output)
        return 16

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_5_output, num_attn_0_4_output):
        key = (num_attn_1_5_output, num_attn_0_4_output)
        return 36

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_7_output, attn_1_7_output):
        if attn_0_7_output in {"1", "4", "0", "<s>"}:
            return attn_1_7_output == ""
        elif attn_0_7_output in {"2"}:
            return attn_1_7_output == "3"
        elif attn_0_7_output in {"3"}:
            return attn_1_7_output == "1"
        elif attn_0_7_output in {"</s>"}:
            return attn_1_7_output == "4"

    attn_2_0_pattern = select_closest(attn_1_7_outputs, attn_0_7_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_0_3_output, token):
        if mlp_0_3_output in {0, 17, 3}:
            return token == "1"
        elif mlp_0_3_output in {
            1,
            2,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            14,
            15,
            16,
            19,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            29,
            30,
            32,
            33,
            34,
            35,
            36,
            39,
        }:
            return token == ""
        elif mlp_0_3_output in {11, 37}:
            return token == "<s>"
        elif mlp_0_3_output in {12, 38, 31}:
            return token == "</s>"
        elif mlp_0_3_output in {13}:
            return token == "4"
        elif mlp_0_3_output in {18}:
            return token == "2"
        elif mlp_0_3_output in {20, 28}:
            return token == "0"

    attn_2_1_pattern = select_closest(tokens, mlp_0_3_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_6_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"2", "3"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "3"
        elif q_token in {"</s>"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == "</s>"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_5_output, mlp_1_0_output):
        if attn_0_5_output in {"1", "0"}:
            return mlp_1_0_output == 37
        elif attn_0_5_output in {"2", "</s>", "<s>"}:
            return mlp_1_0_output == 7
        elif attn_0_5_output in {"3"}:
            return mlp_1_0_output == 22
        elif attn_0_5_output in {"4"}:
            return mlp_1_0_output == 10

    attn_2_3_pattern = select_closest(mlp_1_0_outputs, attn_0_5_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_1_0_output, attn_1_6_output):
        if attn_1_0_output in {"</s>", "3", "0", "2", "4", "1"}:
            return attn_1_6_output == ""
        elif attn_1_0_output in {"<s>"}:
            return attn_1_6_output == "<pad>"

    attn_2_4_pattern = select_closest(attn_1_6_outputs, attn_1_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_5_output, mlp_1_2_output):
        if attn_0_5_output in {"0"}:
            return mlp_1_2_output == 14
        elif attn_0_5_output in {"1"}:
            return mlp_1_2_output == 29
        elif attn_0_5_output in {"2"}:
            return mlp_1_2_output == 26
        elif attn_0_5_output in {"3"}:
            return mlp_1_2_output == 5
        elif attn_0_5_output in {"4"}:
            return mlp_1_2_output == 33
        elif attn_0_5_output in {"</s>"}:
            return mlp_1_2_output == 6
        elif attn_0_5_output in {"<s>"}:
            return mlp_1_2_output == 0

    attn_2_5_pattern = select_closest(mlp_1_2_outputs, attn_0_5_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_2_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_7_output, attn_1_6_output):
        if attn_0_7_output in {"0"}:
            return attn_1_6_output == "<pad>"
        elif attn_0_7_output in {"1", "4", "<s>"}:
            return attn_1_6_output == ""
        elif attn_0_7_output in {"2"}:
            return attn_1_6_output == "1"
        elif attn_0_7_output in {"3"}:
            return attn_1_6_output == "<s>"
        elif attn_0_7_output in {"</s>"}:
            return attn_1_6_output == "</s>"

    attn_2_6_pattern = select_closest(attn_1_6_outputs, attn_0_7_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_7_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_1_6_output, token):
        if attn_1_6_output in {"1", "0", "<s>", "2"}:
            return token == ""
        elif attn_1_6_output in {"3"}:
            return token == "2"
        elif attn_1_6_output in {"4", "</s>"}:
            return token == "</s>"

    attn_2_7_pattern = select_closest(tokens, attn_1_6_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_5_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_3_output, attn_1_0_output):
        if mlp_0_3_output in {0, 32, 31}:
            return attn_1_0_output == "0"
        elif mlp_0_3_output in {1, 5}:
            return attn_1_0_output == "2"
        elif mlp_0_3_output in {
            33,
            2,
            34,
            4,
            35,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            37,
            38,
            39,
            36,
            30,
        }:
            return attn_1_0_output == "1"
        elif mlp_0_3_output in {3}:
            return attn_1_0_output == "<s>"
        elif mlp_0_3_output in {
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
        }:
            return attn_1_0_output == ""
        elif mlp_0_3_output in {29}:
            return attn_1_0_output == "</s>"

    num_attn_2_0_pattern = select(attn_1_0_outputs, mlp_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_1_0_output):
        if position in {
            0,
            2,
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
            16,
            17,
            29,
            30,
            31,
            32,
            33,
            35,
            37,
            38,
            39,
        }:
            return attn_1_0_output == "2"
        elif position in {1, 20, 21}:
            return attn_1_0_output == "</s>"
        elif position in {3}:
            return attn_1_0_output == "3"
        elif position in {15}:
            return attn_1_0_output == "1"
        elif position in {18, 19, 34, 36}:
            return attn_1_0_output == "<s>"
        elif position in {22, 23, 24, 25, 26, 27, 28}:
            return attn_1_0_output == ""

    num_attn_2_1_pattern = select(attn_1_0_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_7_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_5_output, attn_1_0_output):
        if attn_1_5_output in {"</s>", "3", "0", "<s>", "2", "4", "1"}:
            return attn_1_0_output == "3"

    num_attn_2_2_pattern = select(attn_1_0_outputs, attn_1_5_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, attn_1_0_output):
        if position in {
            0,
            32,
            33,
            35,
            36,
            5,
            37,
            7,
            8,
            9,
            10,
            11,
            12,
            38,
            14,
            39,
            30,
            31,
        }:
            return attn_1_0_output == "1"
        elif position in {1}:
            return attn_1_0_output == "</s>"
        elif position in {34, 2, 3, 4, 29}:
            return attn_1_0_output == "2"
        elif position in {6}:
            return attn_1_0_output == "3"
        elif position in {13}:
            return attn_1_0_output == "0"
        elif position in {20, 15}:
            return attn_1_0_output == "<s>"
        elif position in {16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28}:
            return attn_1_0_output == ""

    num_attn_2_3_pattern = select(attn_1_0_outputs, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(num_mlp_0_2_output, token):
        if num_mlp_0_2_output in {
            0,
            2,
            3,
            4,
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
            22,
            24,
            25,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            38,
            39,
        }:
            return token == "1"
        elif num_mlp_0_2_output in {1, 35, 37, 21, 26}:
            return token == "0"
        elif num_mlp_0_2_output in {5}:
            return token == "<s>"
        elif num_mlp_0_2_output in {27, 23}:
            return token == "3"

    num_attn_2_4_pattern = select(tokens, num_mlp_0_2_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(position, attn_1_0_output):
        if position in {0, 32, 38, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}:
            return attn_1_0_output == ""
        elif position in {1, 2}:
            return attn_1_0_output == "2"
        elif position in {34, 3, 4, 36, 6, 7, 8, 9, 10, 11, 37, 39}:
            return attn_1_0_output == "0"
        elif position in {13, 12, 5, 14}:
            return attn_1_0_output == "<s>"
        elif position in {17, 35, 33, 15}:
            return attn_1_0_output == "1"
        elif position in {16, 18, 19}:
            return attn_1_0_output == "</s>"

    num_attn_2_5_pattern = select(attn_1_0_outputs, positions, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_1_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(position, token):
        if position in {
            0,
            11,
            12,
            13,
            14,
            15,
            16,
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
        }:
            return token == ""
        elif position in {1, 39}:
            return token == "</s>"
        elif position in {
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
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
            return token == "1"
        elif position in {18}:
            return token == "<s>"

    num_attn_2_6_pattern = select(tokens, positions, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_6_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 29}:
            return token == "<s>"
        elif mlp_0_1_output in {
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            10,
            14,
            15,
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
            30,
            31,
            34,
            38,
        }:
            return token == ""
        elif mlp_0_1_output in {32, 33, 35, 36, 37, 39, 8, 12, 13, 16, 17}:
            return token == "0"
        elif mlp_0_1_output in {9, 11}:
            return token == "2"

    num_attn_2_7_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_1_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_6_output, attn_2_5_output):
        key = (attn_2_6_output, attn_2_5_output)
        if key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("4", "0"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("</s>", "0"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
        }:
            return 37
        return 31

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_6_outputs, attn_2_5_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, position):
        key = (attn_2_2_output, position)
        if key in {
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 25),
            ("0", 26),
            ("0", 27),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 14),
            ("1", 17),
            ("1", 21),
            ("1", 23),
            ("1", 25),
            ("1", 26),
            ("1", 27),
            ("1", 28),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("2", 14),
            ("2", 16),
            ("2", 17),
            ("2", 20),
            ("2", 21),
            ("2", 23),
            ("2", 25),
            ("2", 26),
            ("2", 27),
            ("2", 28),
            ("2", 37),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 14),
            ("3", 16),
            ("3", 17),
            ("3", 19),
            ("3", 20),
            ("3", 21),
            ("3", 23),
            ("3", 25),
            ("3", 26),
            ("3", 27),
            ("3", 28),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 14),
            ("4", 16),
            ("4", 17),
            ("4", 20),
            ("4", 21),
            ("4", 23),
            ("4", 25),
            ("4", 26),
            ("4", 27),
            ("4", 28),
            ("4", 37),
            ("</s>", 5),
            ("</s>", 6),
            ("</s>", 7),
            ("</s>", 14),
            ("</s>", 16),
            ("</s>", 17),
            ("</s>", 20),
            ("</s>", 21),
            ("</s>", 23),
            ("</s>", 25),
            ("</s>", 26),
            ("</s>", 27),
            ("</s>", 28),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 14),
            ("<s>", 16),
            ("<s>", 17),
            ("<s>", 20),
            ("<s>", 21),
            ("<s>", 23),
            ("<s>", 25),
            ("<s>", 26),
            ("<s>", 27),
            ("<s>", 28),
        }:
            return 18
        elif key in {
            ("0", 21),
            ("1", 16),
            ("1", 20),
            ("2", 19),
            ("4", 19),
            ("<s>", 19),
        }:
            return 10
        return 16

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, positions)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_7_output, attn_2_5_output):
        key = (attn_2_7_output, attn_2_5_output)
        if key in {
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "</s>"),
            ("2", "<s>"),
        }:
            return 24
        elif key in {("0", "4"), ("4", "4")}:
            return 5
        return 20

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_7_outputs, attn_2_5_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_0_5_output, num_mlp_0_1_output):
        key = (attn_0_5_output, num_mlp_0_1_output)
        return 14

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_0_5_outputs, num_mlp_0_1_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_7_output, num_attn_2_2_output):
        key = (num_attn_2_7_output, num_attn_2_2_output)
        return 22

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_3_output, num_attn_1_2_output):
        key = (num_attn_1_3_output, num_attn_1_2_output)
        if key in {(0, 0)}:
            return 0
        elif key in {(1, 0)}:
            return 33
        return 24

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_3_output, num_attn_2_0_output):
        key = (num_attn_2_3_output, num_attn_2_0_output)
        return 9

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_7_output, num_attn_1_3_output):
        key = (num_attn_1_7_output, num_attn_1_3_output)
        return 6

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_3_outputs)
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


print(run(["<s>", "3", "4", "0", "1", "3", "0", "</s>"]))
