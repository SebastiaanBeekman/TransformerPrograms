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
        "output/length/rasp/sort/trainlength40/s2/sort_weights.csv",
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
        if q_position in {0, 24, 22, 23}:
            return k_position == 19
        elif q_position in {1, 2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {5, 6}:
            return k_position == 10
        elif q_position in {42, 7}:
            return k_position == 13
        elif q_position in {8, 11}:
            return k_position == 16
        elif q_position in {9, 10}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 22
        elif q_position in {13, 14, 15}:
            return k_position == 23
        elif q_position in {16, 46}:
            return k_position == 26
        elif q_position in {17, 28, 29, 31}:
            return k_position == 24
        elif q_position in {18, 19}:
            return k_position == 15
        elif q_position in {20, 47, 39}:
            return k_position == 33
        elif q_position in {21}:
            return k_position == 18
        elif q_position in {25, 27}:
            return k_position == 36
        elif q_position in {26}:
            return k_position == 35
        elif q_position in {32, 30}:
            return k_position == 25
        elif q_position in {33}:
            return k_position == 7
        elif q_position in {34, 35, 36}:
            return k_position == 30
        elif q_position in {41, 37, 38}:
            return k_position == 31
        elif q_position in {40}:
            return k_position == 38
        elif q_position in {43}:
            return k_position == 41
        elif q_position in {44}:
            return k_position == 39
        elif q_position in {45}:
            return k_position == 27
        elif q_position in {48}:
            return k_position == 44
        elif q_position in {49}:
            return k_position == 43

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {1, 11}:
            return k_position == 2
        elif q_position in {2, 35}:
            return k_position == 4
        elif q_position in {48, 33, 3}:
            return k_position == 5
        elif q_position in {4, 14}:
            return k_position == 6
        elif q_position in {18, 5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {34, 7}:
            return k_position == 13
        elif q_position in {8, 32, 36}:
            return k_position == 11
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {10, 45, 46}:
            return k_position == 1
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {37, 15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 21
        elif q_position in {20}:
            return k_position == 22
        elif q_position in {21}:
            return k_position == 23
        elif q_position in {49, 22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 26
        elif q_position in {24, 40}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 9
        elif q_position in {26}:
            return k_position == 27
        elif q_position in {27, 28, 47}:
            return k_position == 30
        elif q_position in {29}:
            return k_position == 31
        elif q_position in {30}:
            return k_position == 32
        elif q_position in {39, 31}:
            return k_position == 18
        elif q_position in {38}:
            return k_position == 28
        elif q_position in {41, 44}:
            return k_position == 46
        elif q_position in {42}:
            return k_position == 34
        elif q_position in {43}:
            return k_position == 43

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            10,
            17,
            18,
            25,
            28,
            32,
            34,
            35,
            36,
            38,
            39,
        }:
            return token == "3"
        elif position in {1, 11, 14, 15, 19, 20, 21, 22}:
            return token == "2"
        elif position in {9, 12, 13, 16, 23, 26, 29, 30, 31}:
            return token == "4"
        elif position in {24, 37}:
            return token == "1"
        elif position in {27}:
            return token == "0"
        elif position in {33}:
            return token == "<pad>"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"1", "<s>", "0", "</s>", "3"}:
            return k_token == "3"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "4"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 32, 8, 9, 10, 11, 12, 13, 14, 17, 28, 30}:
            return token == "4"
        elif position in {1, 2, 3, 4, 5, 6, 24}:
            return token == "2"
        elif position in {15, 36, 22, 7}:
            return token == "0"
        elif position in {37, 40, 41, 42, 43, 44, 45, 46, 47, 16, 48, 18, 49}:
            return token == ""
        elif position in {33, 34, 35, 39, 19, 20, 26, 27, 29, 31}:
            return token == "3"
        elif position in {21, 23}:
            return token == "</s>"
        elif position in {25}:
            return token == "1"
        elif position in {38}:
            return token == "<s>"

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_token, k_token):
        if q_token in {"0", "</s>", "3"}:
            return k_token == "3"
        elif q_token in {"1", "<s>"}:
            return k_token == ""
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "4"

    attn_0_5_pattern = select_closest(tokens, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {
            0,
            1,
            2,
            3,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            20,
            22,
            23,
            26,
            27,
            28,
            29,
            30,
            31,
            35,
        }:
            return token == "4"
        elif position in {33, 5, 39}:
            return token == "3"
        elif position in {34, 37, 12, 18, 19, 21, 24}:
            return token == "2"
        elif position in {25, 38}:
            return token == "1"
        elif position in {32}:
            return token == "<s>"
        elif position in {36, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 49}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5, 38}:
            return k_position == 0
        elif q_position in {35, 6}:
            return k_position == 7
        elif q_position in {33, 7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12, 36}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {34, 15}:
            return k_position == 6
        elif q_position in {16, 17, 47, 39}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {19, 44}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {27, 22, 23}:
            return k_position == 29
        elif q_position in {24, 25}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 31
        elif q_position in {29}:
            return k_position == 32
        elif q_position in {30, 31}:
            return k_position == 33
        elif q_position in {32}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 38
        elif q_position in {40}:
            return k_position == 44
        elif q_position in {41}:
            return k_position == 36
        elif q_position in {42}:
            return k_position == 42
        elif q_position in {43}:
            return k_position == 27
        elif q_position in {45}:
            return k_position == 40
        elif q_position in {46}:
            return k_position == 41
        elif q_position in {48}:
            return k_position == 49

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 32
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 16
        elif q_position in {13, 39}:
            return k_position == 17
        elif q_position in {14, 15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 20
        elif q_position in {17}:
            return k_position == 21
        elif q_position in {18}:
            return k_position == 22
        elif q_position in {19}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {21, 22}:
            return k_position == 27
        elif q_position in {23}:
            return k_position == 29
        elif q_position in {24, 26, 44}:
            return k_position == 31
        elif q_position in {25}:
            return k_position == 30
        elif q_position in {27}:
            return k_position == 33
        elif q_position in {28, 29}:
            return k_position == 34
        elif q_position in {30}:
            return k_position == 35
        elif q_position in {31}:
            return k_position == 36
        elif q_position in {32, 40, 47}:
            return k_position == 37
        elif q_position in {33}:
            return k_position == 38
        elif q_position in {34}:
            return k_position == 39
        elif q_position in {42, 35}:
            return k_position == 49
        elif q_position in {36}:
            return k_position == 41
        elif q_position in {41, 45, 37}:
            return k_position == 46
        elif q_position in {48, 38}:
            return k_position == 43
        elif q_position in {43}:
            return k_position == 28
        elif q_position in {46}:
            return k_position == 42
        elif q_position in {49}:
            return k_position == 45

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 32, 35, 5, 6, 7, 8, 9, 38, 31}:
            return token == "<s>"
        elif position in {1, 2, 33, 36, 37, 40, 41, 42, 44, 45, 46, 47, 48, 26, 27, 29}:
            return token == ""
        elif position in {3, 4}:
            return token == "<pad>"
        elif position in {10, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 28}:
            return token == "0"
        elif position in {11, 12, 16, 49, 20}:
            return token == "1"
        elif position in {25, 34, 30, 39}:
            return token == "</s>"
        elif position in {43}:
            return token == "2"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 32, 34, 35, 36, 38, 39, 27, 29, 31}:
            return token == "0"
        elif position in {1, 2, 3, 4, 40, 41, 42, 43, 44, 45, 47, 49}:
            return token == ""
        elif position in {48, 5, 46}:
            return token == "<pad>"
        elif position in {6, 8, 9, 10, 11, 12, 14, 15, 16, 18}:
            return token == "</s>"
        elif position in {17, 19, 13, 7}:
            return token == "<s>"
        elif position in {33, 37, 20, 26, 28, 30}:
            return token == "1"
        elif position in {21, 22, 23, 24, 25}:
            return token == "4"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 2, 3, 4, 5, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}:
            return token == ""
        elif position in {6, 7, 8, 9, 14}:
            return token == "</s>"
        elif position in {10, 11, 12, 13, 15}:
            return token == "<s>"
        elif position in {16, 17, 18, 19, 20, 49}:
            return token == "4"
        elif position in {33, 34, 36, 37, 21, 22, 23, 24, 25, 26, 28}:
            return token == "0"
        elif position in {32, 35, 38, 27, 29, 30, 31}:
            return token == "1"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 39}:
            return token == "0"
        elif position in {1}:
            return token == "</s>"
        elif position in {2, 3, 4, 5, 6, 7, 8, 9, 10, 40, 42, 44, 45, 46, 47, 48, 49}:
            return token == "1"
        elif position in {
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
        elif position in {41, 43}:
            return token == "2"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 1, 2, 37, 39}:
            return token == "0"
        elif position in {3, 4, 5, 6, 7, 8, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {9, 10, 11, 14, 15, 17, 18, 22, 23}:
            return token == "</s>"
        elif position in {12, 13, 16, 19, 20, 21, 24}:
            return token == "<s>"
        elif position in {32, 33, 34, 35, 36, 38, 25}:
            return token == "1"
        elif position in {26, 27, 28, 29, 30, 31}:
            return token == "4"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 1, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45, 47, 48, 49}:
            return token == ""
        elif position in {11, 6, 14}:
            return token == "</s>"
        elif position in {7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21}:
            return token == "<s>"
        elif position in {34, 36, 37, 38, 22, 23, 24, 26, 27, 28, 29, 31}:
            return token == "1"
        elif position in {33, 35, 39, 25, 30}:
            return token == "0"
        elif position in {32}:
            return token == "2"
        elif position in {46}:
            return token == "<pad>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1}:
            return token == "</s>"
        elif position in {2, 3, 4, 5, 6, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
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
        elif position in {40}:
            return token == "1"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        if key in {
            ("0", 1),
            ("0", 3),
            ("0", 4),
            ("0", 40),
            ("0", 41),
            ("0", 42),
            ("0", 43),
            ("0", 44),
            ("0", 45),
            ("0", 46),
            ("0", 47),
            ("0", 48),
            ("0", 49),
            ("1", 3),
            ("1", 4),
            ("2", 1),
            ("2", 3),
            ("2", 4),
            ("2", 40),
            ("2", 41),
            ("2", 42),
            ("2", 43),
            ("2", 44),
            ("2", 45),
            ("2", 46),
            ("2", 47),
            ("2", 48),
            ("2", 49),
            ("3", 3),
            ("3", 4),
            ("4", 3),
            ("4", 4),
            ("</s>", 3),
            ("</s>", 4),
            ("<s>", 3),
            ("<s>", 4),
        }:
            return 17
        elif key in {
            ("0", 2),
            ("1", 1),
            ("1", 2),
            ("2", 2),
            ("3", 1),
            ("3", 2),
            ("4", 1),
            ("4", 2),
            ("</s>", 1),
            ("</s>", 2),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 37
        return 42

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_4_output):
        key = (attn_0_1_output, attn_0_4_output)
        return 45

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_4_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_1_output, attn_0_2_output):
        key = (attn_0_1_output, attn_0_2_output)
        return 7

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_2_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position, attn_0_4_output):
        key = (position, attn_0_4_output)
        if key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "</s>"),
            (0, "<s>"),
            (1, "1"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "1"),
            (3, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "</s>"),
            (4, "<s>"),
            (17, "0"),
            (17, "1"),
            (17, "2"),
            (17, "3"),
            (17, "4"),
            (17, "</s>"),
            (17, "<s>"),
            (22, "0"),
            (22, "1"),
            (22, "2"),
            (22, "3"),
            (22, "4"),
            (22, "</s>"),
            (22, "<s>"),
            (23, "0"),
            (23, "1"),
            (23, "2"),
            (23, "3"),
            (23, "4"),
            (23, "</s>"),
            (23, "<s>"),
            (24, "0"),
            (24, "1"),
            (24, "2"),
            (24, "3"),
            (24, "4"),
            (24, "<s>"),
            (25, "1"),
            (25, "<s>"),
            (31, "0"),
            (31, "1"),
            (31, "2"),
            (31, "3"),
            (31, "4"),
            (31, "</s>"),
            (31, "<s>"),
            (34, "0"),
            (34, "1"),
            (34, "2"),
            (34, "3"),
            (34, "4"),
            (34, "</s>"),
            (34, "<s>"),
            (36, "0"),
            (36, "1"),
            (36, "2"),
            (36, "3"),
            (36, "4"),
            (36, "<s>"),
            (37, "0"),
            (37, "1"),
            (37, "2"),
            (37, "3"),
            (37, "4"),
            (37, "</s>"),
            (37, "<s>"),
            (38, "0"),
            (38, "1"),
            (38, "4"),
            (38, "<s>"),
            (40, "1"),
            (40, "<s>"),
            (41, "1"),
            (41, "<s>"),
            (42, "1"),
            (42, "<s>"),
            (43, "1"),
            (43, "<s>"),
            (44, "1"),
            (44, "<s>"),
            (45, "1"),
            (45, "<s>"),
            (46, "1"),
            (46, "<s>"),
            (47, "1"),
            (47, "<s>"),
            (48, "1"),
            (48, "<s>"),
            (49, "1"),
            (49, "<s>"),
        }:
            return 37
        elif key in {
            (15, "0"),
            (15, "3"),
            (15, "</s>"),
            (24, "</s>"),
            (28, "</s>"),
            (29, "</s>"),
        }:
            return 26
        return 17

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(positions, attn_0_4_outputs)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 27

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_7_output):
        key = (num_attn_0_0_output, num_attn_0_7_output)
        return 6

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_7_output, num_attn_0_5_output):
        key = (num_attn_0_7_output, num_attn_0_5_output)
        return 49

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_4_output, num_attn_0_6_output):
        key = (num_attn_0_4_output, num_attn_0_6_output)
        return 28

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 33, 3, 4, 5, 6, 7, 35, 36, 38, 11, 13, 15, 16, 21}:
            return token == "2"
        elif position in {1, 2, 44, 45, 46}:
            return token == "0"
        elif position in {34, 8, 9, 10, 12, 14, 17, 19, 20, 22, 27, 29}:
            return token == "4"
        elif position in {37, 39, 18, 23, 24, 26, 31}:
            return token == "3"
        elif position in {25}:
            return token == "1"
        elif position in {49, 28}:
            return token == "</s>"
        elif position in {32, 42, 30}:
            return token == "<s>"
        elif position in {40, 41, 43, 47, 48}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, num_mlp_0_0_output):
        if position in {0, 34, 37, 6, 7, 8, 9, 10, 12, 19, 31}:
            return num_mlp_0_0_output == 4
        elif position in {1, 2, 3, 40, 47, 20}:
            return num_mlp_0_0_output == 1
        elif position in {4, 39}:
            return num_mlp_0_0_output == 3
        elif position in {13, 21, 5}:
            return num_mlp_0_0_output == 8
        elif position in {24, 11}:
            return num_mlp_0_0_output == 27
        elif position in {17, 14}:
            return num_mlp_0_0_output == 49
        elif position in {15}:
            return num_mlp_0_0_output == 14
        elif position in {35, 36, 38, 42, 16, 22, 29, 30}:
            return num_mlp_0_0_output == 7
        elif position in {18}:
            return num_mlp_0_0_output == 17
        elif position in {32, 33, 43, 45, 23, 28}:
            return num_mlp_0_0_output == 6
        elif position in {25}:
            return num_mlp_0_0_output == 26
        elif position in {49, 26}:
            return num_mlp_0_0_output == 12
        elif position in {27}:
            return num_mlp_0_0_output == 41
        elif position in {41}:
            return num_mlp_0_0_output == 9
        elif position in {44}:
            return num_mlp_0_0_output == 25
        elif position in {46}:
            return num_mlp_0_0_output == 24
        elif position in {48}:
            return num_mlp_0_0_output == 37

    attn_1_1_pattern = select_closest(num_mlp_0_0_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_5_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 2
        elif q_position in {2, 38}:
            return k_position == 1
        elif q_position in {3, 4}:
            return k_position == 8
        elif q_position in {35, 44, 5}:
            return k_position == 12
        elif q_position in {6}:
            return k_position == 16
        elif q_position in {40, 7}:
            return k_position == 14
        elif q_position in {8, 16, 31}:
            return k_position == 9
        elif q_position in {9, 15}:
            return k_position == 4
        elif q_position in {10}:
            return k_position == 20
        elif q_position in {32, 34, 36, 11, 14}:
            return k_position == 15
        elif q_position in {25, 29, 12, 21}:
            return k_position == 10
        elif q_position in {33, 26, 37, 13}:
            return k_position == 11
        elif q_position in {17}:
            return k_position == 23
        elif q_position in {18, 19}:
            return k_position == 33
        elif q_position in {20}:
            return k_position == 18
        elif q_position in {22}:
            return k_position == 39
        elif q_position in {24, 27, 23}:
            return k_position == 0
        elif q_position in {28}:
            return k_position == 26
        elif q_position in {30}:
            return k_position == 28
        elif q_position in {42, 47, 39}:
            return k_position == 7
        elif q_position in {41}:
            return k_position == 37
        elif q_position in {43}:
            return k_position == 21
        elif q_position in {45}:
            return k_position == 36
        elif q_position in {46}:
            return k_position == 31
        elif q_position in {48}:
            return k_position == 17
        elif q_position in {49}:
            return k_position == 24

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_6_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_3_output, attn_0_1_output):
        if mlp_0_3_output in {0, 13, 15, 49, 24, 28}:
            return attn_0_1_output == "1"
        elif mlp_0_3_output in {
            32,
            1,
            2,
            3,
            4,
            5,
            38,
            8,
            40,
            10,
            42,
            43,
            44,
            20,
            22,
            23,
            25,
            29,
        }:
            return attn_0_1_output == ""
        elif mlp_0_3_output in {11, 6}:
            return attn_0_1_output == "</s>"
        elif mlp_0_3_output in {41, 7}:
            return attn_0_1_output == "<s>"
        elif mlp_0_3_output in {9, 12, 14, 18, 21}:
            return attn_0_1_output == "4"
        elif mlp_0_3_output in {34, 35, 39, 45, 16, 30, 31}:
            return attn_0_1_output == "3"
        elif mlp_0_3_output in {33, 36, 17, 19, 26}:
            return attn_0_1_output == "2"
        elif mlp_0_3_output in {37, 46, 47, 48, 27}:
            return attn_0_1_output == "0"

    attn_1_3_pattern = select_closest(attn_0_1_outputs, mlp_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3, 13}:
            return k_position == 22
        elif q_position in {40, 4, 5, 38}:
            return k_position == 3
        elif q_position in {16, 12, 6}:
            return k_position == 8
        elif q_position in {10, 42, 46, 7}:
            return k_position == 6
        elif q_position in {8, 32, 14}:
            return k_position == 10
        elif q_position in {9, 39}:
            return k_position == 4
        elif q_position in {24, 11}:
            return k_position == 32
        elif q_position in {15}:
            return k_position == 20
        elif q_position in {17, 18, 20, 33}:
            return k_position == 11
        elif q_position in {19, 47}:
            return k_position == 12
        elif q_position in {21}:
            return k_position == 44
        elif q_position in {27, 22}:
            return k_position == 9
        elif q_position in {25, 36, 23}:
            return k_position == 7
        elif q_position in {26, 35}:
            return k_position == 17
        elif q_position in {28}:
            return k_position == 37
        elif q_position in {29}:
            return k_position == 23
        elif q_position in {30}:
            return k_position == 21
        elif q_position in {31}:
            return k_position == 14
        elif q_position in {34}:
            return k_position == 33
        elif q_position in {37}:
            return k_position == 24
        elif q_position in {48, 41}:
            return k_position == 0
        elif q_position in {43}:
            return k_position == 34
        elif q_position in {44}:
            return k_position == 19
        elif q_position in {45}:
            return k_position == 18
        elif q_position in {49}:
            return k_position == 42

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, num_mlp_0_3_output):
        if position in {0, 16, 42}:
            return num_mlp_0_3_output == 22
        elif position in {1, 43, 45, 48, 49}:
            return num_mlp_0_3_output == 1
        elif position in {24, 35, 2, 3}:
            return num_mlp_0_3_output == 6
        elif position in {4, 37, 39, 8, 10, 12, 14}:
            return num_mlp_0_3_output == 4
        elif position in {5, 6, 9, 15, 18, 19, 22}:
            return num_mlp_0_3_output == 3
        elif position in {33, 7}:
            return num_mlp_0_3_output == 5
        elif position in {25, 11}:
            return num_mlp_0_3_output == 12
        elif position in {13}:
            return num_mlp_0_3_output == 40
        elif position in {17}:
            return num_mlp_0_3_output == 28
        elif position in {20}:
            return num_mlp_0_3_output == 34
        elif position in {21}:
            return num_mlp_0_3_output == 20
        elif position in {29, 23}:
            return num_mlp_0_3_output == 37
        elif position in {26}:
            return num_mlp_0_3_output == 18
        elif position in {27}:
            return num_mlp_0_3_output == 24
        elif position in {28}:
            return num_mlp_0_3_output == 41
        elif position in {30}:
            return num_mlp_0_3_output == 45
        elif position in {32, 34, 31}:
            return num_mlp_0_3_output == 7
        elif position in {40, 36}:
            return num_mlp_0_3_output == 46
        elif position in {38}:
            return num_mlp_0_3_output == 16
        elif position in {41}:
            return num_mlp_0_3_output == 26
        elif position in {44}:
            return num_mlp_0_3_output == 31
        elif position in {46}:
            return num_mlp_0_3_output == 38
        elif position in {47}:
            return num_mlp_0_3_output == 44

    attn_1_5_pattern = select_closest(num_mlp_0_3_outputs, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, mlp_0_0_output):
        if position in {0, 37, 6, 11, 14, 15, 20, 21, 22}:
            return mlp_0_0_output == 4
        elif position in {1, 41}:
            return mlp_0_0_output == 1
        elif position in {2, 3, 5, 8, 9, 12, 17, 29}:
            return mlp_0_0_output == 37
        elif position in {4, 38, 39}:
            return mlp_0_0_output == 3
        elif position in {7, 10, 16, 18, 19, 26}:
            return mlp_0_0_output == 17
        elif position in {33, 34, 13, 23, 30}:
            return mlp_0_0_output == 5
        elif position in {24, 25}:
            return mlp_0_0_output == 6
        elif position in {32, 35, 27, 31}:
            return mlp_0_0_output == 7
        elif position in {28}:
            return mlp_0_0_output == 42
        elif position in {36}:
            return mlp_0_0_output == 18
        elif position in {40}:
            return mlp_0_0_output == 32
        elif position in {42, 45, 46, 47, 49}:
            return mlp_0_0_output == 0
        elif position in {43}:
            return mlp_0_0_output == 13
        elif position in {44}:
            return mlp_0_0_output == 39
        elif position in {48}:
            return mlp_0_0_output == 10

    attn_1_6_pattern = select_closest(mlp_0_0_outputs, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_3_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, token):
        if position in {0, 3, 6, 12, 13, 27}:
            return token == "4"
        elif position in {1, 2, 5, 17, 25, 26, 29, 30}:
            return token == "1"
        elif position in {35, 4, 7, 10, 11, 14, 15, 18, 19, 23}:
            return token == "2"
        elif position in {34, 8, 9, 16, 22, 24, 28, 31}:
            return token == "3"
        elif position in {20}:
            return token == "0"
        elif position in {40, 42, 21}:
            return token == "</s>"
        elif position in {32, 33, 36, 38, 39}:
            return token == "<s>"
        elif position in {37, 41, 43, 47, 48}:
            return token == ""
        elif position in {49, 44, 45, 46}:
            return token == "<pad>"

    attn_1_7_pattern = select_closest(tokens, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_1_output):
        if position in {
            0,
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
            39,
            41,
            42,
            43,
            44,
            48,
            49,
        }:
            return attn_0_1_output == ""
        elif position in {1, 30, 31}:
            return attn_0_1_output == "</s>"
        elif position in {2, 3, 37, 38}:
            return attn_0_1_output == "2"
        elif position in {4, 7}:
            return attn_0_1_output == "0"
        elif position in {5, 6, 40, 45, 46, 47}:
            return attn_0_1_output == "1"
        elif position in {32, 33, 34, 35, 36}:
            return attn_0_1_output == "4"

    num_attn_1_0_pattern = select(attn_0_1_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_5_output, position):
        if attn_0_5_output in {"0", "</s>", "4"}:
            return position == 32
        elif attn_0_5_output in {"1"}:
            return position == 17
        elif attn_0_5_output in {"2"}:
            return position == 4
        elif attn_0_5_output in {"3"}:
            return position == 37
        elif attn_0_5_output in {"<s>"}:
            return position == 34

    num_attn_1_1_pattern = select(positions, attn_0_5_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_6_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 43, 12}:
            return k_position == 20
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3, 4}:
            return k_position == 14
        elif q_position in {34, 5, 11, 13, 14}:
            return k_position == 19
        elif q_position in {33, 27, 6, 47}:
            return k_position == 13
        elif q_position in {8, 9, 39, 7}:
            return k_position == 1
        elif q_position in {41, 10}:
            return k_position == 22
        elif q_position in {15}:
            return k_position == 21
        elif q_position in {16}:
            return k_position == 16
        elif q_position in {17}:
            return k_position == 27
        elif q_position in {18}:
            return k_position == 25
        elif q_position in {42, 49, 19, 22, 23, 24, 26}:
            return k_position == 33
        elif q_position in {25, 20, 21}:
            return k_position == 32
        elif q_position in {28}:
            return k_position == 36
        elif q_position in {48, 29}:
            return k_position == 37
        elif q_position in {35, 30}:
            return k_position == 38
        elif q_position in {31}:
            return k_position == 39
        elif q_position in {32}:
            return k_position == 35
        elif q_position in {36, 45}:
            return k_position == 8
        elif q_position in {44, 37}:
            return k_position == 6
        elif q_position in {38}:
            return k_position == 9
        elif q_position in {40}:
            return k_position == 17
        elif q_position in {46}:
            return k_position == 12

    num_attn_1_2_pattern = select(positions, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_5_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_2_output):
        if position in {0}:
            return attn_0_2_output == "0"
        elif position in {1, 18, 3}:
            return attn_0_2_output == "<s>"
        elif position in {2, 6, 7, 8, 9, 10, 12, 45}:
            return attn_0_2_output == "2"
        elif position in {4, 46, 17, 22, 23, 25}:
            return attn_0_2_output == "</s>"
        elif position in {
            5,
            15,
            19,
            20,
            21,
            24,
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
            43,
            44,
        }:
            return attn_0_2_output == ""
        elif position in {39, 42, 11, 13, 14, 47, 16, 48, 49}:
            return attn_0_2_output == "1"

    num_attn_1_3_pattern = select(attn_0_2_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_5_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_0_output, attn_0_4_output):
        if attn_0_0_output in {0, 7, 41, 47, 49}:
            return attn_0_4_output == "0"
        elif attn_0_0_output in {1, 2, 3, 4, 5, 6, 12, 15, 16, 17}:
            return attn_0_4_output == "</s>"
        elif attn_0_0_output in {8, 40, 43, 11, 45, 14, 48}:
            return attn_0_4_output == "1"
        elif attn_0_0_output in {
            9,
            10,
            13,
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
            35,
            36,
            37,
            38,
            39,
            42,
            44,
            46,
        }:
            return attn_0_4_output == ""
        elif attn_0_0_output in {27}:
            return attn_0_4_output == "<pad>"

    num_attn_1_4_pattern = select(attn_0_4_outputs, attn_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_6_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, token):
        if position in {
            0,
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
            46,
        }:
            return token == ""
        elif position in {1, 12, 13}:
            return token == "</s>"
        elif position in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 41, 42, 43, 45, 48, 49}:
            return token == "0"
        elif position in {39}:
            return token == "<s>"
        elif position in {40, 44, 47}:
            return token == "1"

    num_attn_1_5_pattern = select(tokens, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_1_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_7_output):
        if position in {
            0,
            6,
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
            41,
            42,
            45,
        }:
            return attn_0_7_output == ""
        elif position in {1, 2, 3}:
            return attn_0_7_output == "<s>"
        elif position in {4, 5, 43, 46, 47, 48, 49}:
            return attn_0_7_output == "1"
        elif position in {39, 7}:
            return attn_0_7_output == "0"
        elif position in {40}:
            return attn_0_7_output == "<pad>"
        elif position in {44}:
            return attn_0_7_output == "4"

    num_attn_1_6_pattern = select(attn_0_7_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_4_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 37
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2, 3}:
            return k_position == 7
        elif q_position in {4, 6, 39, 40, 41, 44, 45, 46, 48}:
            return k_position == 1
        elif q_position in {5, 7}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {9, 10}:
            return k_position == 14
        elif q_position in {11, 12}:
            return k_position == 18
        elif q_position in {47, 13, 14, 15}:
            return k_position == 20
        elif q_position in {16}:
            return k_position == 22
        elif q_position in {17, 18, 49}:
            return k_position == 25
        elif q_position in {26, 19, 20, 21}:
            return k_position == 30
        elif q_position in {25, 43, 22}:
            return k_position == 31
        elif q_position in {42, 27, 23}:
            return k_position == 32
        elif q_position in {24}:
            return k_position == 34
        elif q_position in {28}:
            return k_position == 38
        elif q_position in {32, 35, 29, 30}:
            return k_position == 39
        elif q_position in {31}:
            return k_position == 46
        elif q_position in {33}:
            return k_position == 48
        elif q_position in {34}:
            return k_position == 40
        elif q_position in {36}:
            return k_position == 41
        elif q_position in {37}:
            return k_position == 49
        elif q_position in {38}:
            return k_position == 43

    num_attn_1_7_pattern = select(positions, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_1_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(num_mlp_0_3_output, attn_1_5_output):
        key = (num_mlp_0_3_output, attn_1_5_output)
        if key in {
            (0, "<s>"),
            (2, "<s>"),
            (6, "</s>"),
            (6, "<s>"),
            (7, "<s>"),
            (11, "<s>"),
            (17, "</s>"),
            (17, "<s>"),
            (18, "1"),
            (18, "2"),
            (18, "4"),
            (18, "</s>"),
            (18, "<s>"),
            (27, "<s>"),
            (29, "<s>"),
            (30, "<s>"),
            (32, "<s>"),
            (38, "</s>"),
            (38, "<s>"),
            (40, "</s>"),
            (40, "<s>"),
            (41, "<s>"),
        }:
            return 19
        elif key in {
            (49, "0"),
            (49, "1"),
            (49, "2"),
            (49, "3"),
            (49, "4"),
            (49, "</s>"),
            (49, "<s>"),
        }:
            return 22
        return 1

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, attn_1_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(mlp_0_0_output, attn_0_7_output):
        key = (mlp_0_0_output, attn_0_7_output)
        return 29

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, attn_0_7_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(mlp_0_0_output, num_mlp_0_3_output):
        key = (mlp_0_0_output, num_mlp_0_3_output)
        if key in {
            (0, 3),
            (0, 7),
            (0, 12),
            (0, 13),
            (0, 17),
            (0, 30),
            (0, 34),
            (0, 37),
            (1, 30),
            (1, 34),
            (1, 37),
            (2, 0),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 37),
            (2, 38),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 48),
            (2, 49),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 10),
            (4, 12),
            (4, 13),
            (4, 15),
            (4, 17),
            (4, 23),
            (4, 24),
            (4, 27),
            (4, 29),
            (4, 30),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 37),
            (4, 38),
            (4, 41),
            (4, 43),
            (4, 45),
            (4, 47),
            (4, 48),
            (4, 49),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 48),
            (6, 49),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 15),
            (7, 17),
            (7, 23),
            (7, 24),
            (7, 27),
            (7, 29),
            (7, 30),
            (7, 31),
            (7, 33),
            (7, 34),
            (7, 35),
            (7, 37),
            (7, 38),
            (7, 41),
            (7, 43),
            (7, 45),
            (7, 46),
            (7, 47),
            (7, 48),
            (7, 49),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 10),
            (10, 12),
            (10, 13),
            (10, 15),
            (10, 17),
            (10, 23),
            (10, 24),
            (10, 27),
            (10, 29),
            (10, 30),
            (10, 33),
            (10, 34),
            (10, 35),
            (10, 37),
            (10, 38),
            (10, 43),
            (10, 47),
            (10, 48),
            (11, 3),
            (11, 5),
            (11, 7),
            (11, 12),
            (11, 13),
            (11, 17),
            (11, 24),
            (11, 29),
            (11, 30),
            (11, 34),
            (11, 37),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 10),
            (12, 12),
            (12, 13),
            (12, 15),
            (12, 17),
            (12, 23),
            (12, 24),
            (12, 27),
            (12, 29),
            (12, 30),
            (12, 33),
            (12, 34),
            (12, 35),
            (12, 37),
            (12, 38),
            (12, 43),
            (12, 47),
            (12, 48),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 10),
            (13, 12),
            (13, 13),
            (13, 15),
            (13, 17),
            (13, 23),
            (13, 24),
            (13, 27),
            (13, 29),
            (13, 30),
            (13, 33),
            (13, 34),
            (13, 35),
            (13, 37),
            (13, 38),
            (13, 43),
            (13, 47),
            (13, 48),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 10),
            (15, 12),
            (15, 13),
            (15, 17),
            (15, 23),
            (15, 24),
            (15, 29),
            (15, 30),
            (15, 33),
            (15, 34),
            (15, 35),
            (15, 37),
            (15, 38),
            (15, 43),
            (15, 47),
            (15, 48),
            (16, 30),
            (16, 34),
            (16, 37),
            (18, 30),
            (18, 37),
            (19, 30),
            (19, 34),
            (19, 37),
            (20, 12),
            (20, 30),
            (20, 34),
            (20, 37),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 10),
            (21, 12),
            (21, 13),
            (21, 17),
            (21, 23),
            (21, 24),
            (21, 27),
            (21, 29),
            (21, 30),
            (21, 33),
            (21, 34),
            (21, 35),
            (21, 37),
            (21, 38),
            (21, 43),
            (21, 47),
            (21, 48),
            (22, 3),
            (22, 5),
            (22, 7),
            (22, 12),
            (22, 13),
            (22, 17),
            (22, 24),
            (22, 29),
            (22, 30),
            (22, 34),
            (22, 37),
            (22, 47),
            (23, 3),
            (23, 5),
            (23, 7),
            (23, 12),
            (23, 13),
            (23, 17),
            (23, 24),
            (23, 29),
            (23, 30),
            (23, 34),
            (23, 37),
            (23, 47),
            (25, 3),
            (25, 5),
            (25, 7),
            (25, 12),
            (25, 13),
            (25, 17),
            (25, 24),
            (25, 29),
            (25, 30),
            (25, 34),
            (25, 37),
            (25, 47),
            (26, 30),
            (26, 34),
            (26, 37),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 10),
            (30, 12),
            (30, 13),
            (30, 17),
            (30, 23),
            (30, 24),
            (30, 29),
            (30, 30),
            (30, 33),
            (30, 34),
            (30, 35),
            (30, 37),
            (30, 38),
            (30, 43),
            (30, 47),
            (30, 48),
            (31, 30),
            (33, 12),
            (33, 13),
            (33, 17),
            (33, 30),
            (33, 34),
            (33, 37),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 10),
            (34, 12),
            (34, 13),
            (34, 17),
            (34, 23),
            (34, 24),
            (34, 29),
            (34, 30),
            (34, 33),
            (34, 34),
            (34, 37),
            (34, 38),
            (34, 43),
            (34, 47),
            (34, 48),
            (35, 12),
            (35, 30),
            (35, 34),
            (35, 37),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 10),
            (37, 12),
            (37, 13),
            (37, 15),
            (37, 17),
            (37, 23),
            (37, 24),
            (37, 27),
            (37, 29),
            (37, 30),
            (37, 33),
            (37, 34),
            (37, 35),
            (37, 37),
            (37, 38),
            (37, 41),
            (37, 43),
            (37, 45),
            (37, 47),
            (37, 48),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 7),
            (38, 12),
            (38, 13),
            (38, 17),
            (38, 24),
            (38, 29),
            (38, 30),
            (38, 33),
            (38, 34),
            (38, 37),
            (38, 38),
            (38, 47),
            (39, 30),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 10),
            (40, 12),
            (40, 13),
            (40, 17),
            (40, 23),
            (40, 24),
            (40, 29),
            (40, 30),
            (40, 33),
            (40, 34),
            (40, 37),
            (40, 38),
            (40, 43),
            (40, 47),
            (40, 48),
            (41, 30),
            (41, 34),
            (41, 37),
            (43, 30),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 7),
            (44, 12),
            (44, 13),
            (44, 17),
            (44, 24),
            (44, 29),
            (44, 30),
            (44, 34),
            (44, 37),
            (44, 47),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 7),
            (45, 12),
            (45, 13),
            (45, 17),
            (45, 23),
            (45, 24),
            (45, 29),
            (45, 30),
            (45, 33),
            (45, 34),
            (45, 37),
            (45, 38),
            (45, 47),
            (46, 12),
            (46, 30),
            (46, 34),
            (46, 37),
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
            (47, 22),
            (47, 23),
            (47, 24),
            (47, 25),
            (47, 26),
            (47, 27),
            (47, 29),
            (47, 30),
            (47, 31),
            (47, 32),
            (47, 33),
            (47, 34),
            (47, 35),
            (47, 36),
            (47, 37),
            (47, 38),
            (47, 39),
            (47, 40),
            (47, 41),
            (47, 42),
            (47, 43),
            (47, 44),
            (47, 45),
            (47, 46),
            (47, 47),
            (47, 48),
            (47, 49),
        }:
            return 20
        elif key in {
            (0, 5),
            (0, 24),
            (0, 29),
            (0, 47),
            (2, 36),
            (4, 31),
            (4, 46),
            (7, 19),
            (7, 26),
            (7, 32),
            (9, 27),
            (9, 30),
            (11, 4),
            (11, 33),
            (11, 47),
            (12, 45),
            (13, 45),
            (15, 15),
            (15, 27),
            (16, 12),
            (18, 22),
            (18, 25),
            (18, 27),
            (18, 34),
            (19, 12),
            (21, 15),
            (22, 4),
            (22, 33),
            (23, 4),
            (23, 33),
            (23, 38),
            (24, 2),
            (24, 4),
            (24, 5),
            (24, 16),
            (24, 18),
            (24, 22),
            (24, 24),
            (24, 25),
            (24, 27),
            (24, 31),
            (24, 41),
            (24, 44),
            (25, 4),
            (25, 33),
            (26, 12),
            (26, 22),
            (26, 27),
            (27, 30),
            (28, 30),
            (29, 30),
            (30, 15),
            (30, 27),
            (32, 30),
            (33, 3),
            (33, 7),
            (34, 15),
            (34, 27),
            (34, 35),
            (35, 3),
            (35, 7),
            (35, 13),
            (35, 17),
            (36, 27),
            (36, 30),
            (37, 49),
            (38, 23),
            (39, 37),
            (40, 15),
            (40, 27),
            (40, 35),
            (43, 34),
            (43, 37),
            (44, 33),
            (44, 38),
            (45, 6),
            (45, 27),
            (45, 48),
            (46, 3),
            (46, 7),
            (46, 13),
            (46, 17),
            (46, 27),
            (48, 2),
            (48, 4),
            (48, 5),
            (48, 6),
            (48, 8),
            (48, 14),
            (48, 16),
            (48, 18),
            (48, 19),
            (48, 20),
            (48, 21),
            (48, 22),
            (48, 23),
            (48, 24),
            (48, 25),
            (48, 26),
            (48, 27),
            (48, 30),
            (48, 31),
            (48, 32),
            (48, 36),
            (48, 38),
            (48, 39),
            (48, 40),
            (48, 41),
            (48, 44),
            (48, 46),
            (48, 48),
            (49, 27),
        }:
            return 1
        return 35

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, num_mlp_0_3_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(num_mlp_0_1_output, mlp_0_3_output):
        key = (num_mlp_0_1_output, mlp_0_3_output)
        if key in {
            (0, 0),
            (0, 10),
            (0, 18),
            (0, 26),
            (0, 31),
            (1, 0),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 10),
            (1, 18),
            (1, 19),
            (1, 21),
            (1, 24),
            (1, 26),
            (1, 31),
            (1, 33),
            (1, 40),
            (1, 42),
            (1, 44),
            (1, 48),
            (2, 0),
            (2, 2),
            (2, 3),
            (2, 10),
            (2, 18),
            (2, 21),
            (2, 26),
            (2, 31),
            (2, 48),
            (3, 0),
            (3, 2),
            (3, 3),
            (3, 10),
            (3, 18),
            (3, 21),
            (3, 26),
            (3, 31),
            (4, 0),
            (4, 10),
            (4, 26),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 10),
            (5, 18),
            (5, 21),
            (5, 26),
            (5, 31),
            (5, 33),
            (5, 40),
            (5, 48),
            (6, 0),
            (6, 2),
            (6, 3),
            (6, 10),
            (6, 18),
            (6, 21),
            (6, 26),
            (6, 31),
            (6, 48),
            (7, 0),
            (7, 2),
            (7, 3),
            (7, 6),
            (7, 10),
            (7, 18),
            (7, 19),
            (7, 21),
            (7, 24),
            (7, 26),
            (7, 31),
            (7, 33),
            (7, 40),
            (7, 42),
            (7, 48),
            (8, 0),
            (8, 2),
            (8, 10),
            (8, 18),
            (8, 21),
            (8, 26),
            (8, 31),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 10),
            (9, 14),
            (9, 15),
            (9, 18),
            (9, 19),
            (9, 21),
            (9, 23),
            (9, 24),
            (9, 26),
            (9, 31),
            (9, 33),
            (9, 34),
            (9, 38),
            (9, 40),
            (9, 41),
            (9, 42),
            (9, 43),
            (9, 44),
            (9, 45),
            (9, 48),
            (10, 0),
            (10, 2),
            (10, 3),
            (10, 10),
            (10, 18),
            (10, 19),
            (10, 21),
            (10, 26),
            (10, 31),
            (10, 33),
            (10, 40),
            (10, 42),
            (10, 48),
            (11, 0),
            (11, 2),
            (11, 3),
            (11, 10),
            (11, 18),
            (11, 21),
            (11, 26),
            (11, 31),
            (11, 33),
            (11, 40),
            (11, 42),
            (11, 48),
            (12, 0),
            (12, 2),
            (12, 3),
            (12, 10),
            (12, 18),
            (12, 21),
            (12, 26),
            (12, 31),
            (13, 0),
            (13, 2),
            (13, 10),
            (13, 26),
            (13, 31),
            (14, 0),
            (14, 2),
            (14, 3),
            (14, 10),
            (14, 18),
            (14, 21),
            (14, 26),
            (14, 31),
            (14, 33),
            (14, 40),
            (14, 48),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 6),
            (15, 10),
            (15, 14),
            (15, 18),
            (15, 19),
            (15, 21),
            (15, 23),
            (15, 24),
            (15, 26),
            (15, 31),
            (15, 33),
            (15, 40),
            (15, 42),
            (15, 44),
            (15, 45),
            (15, 48),
            (16, 0),
            (16, 2),
            (16, 3),
            (16, 6),
            (16, 10),
            (16, 18),
            (16, 19),
            (16, 21),
            (16, 24),
            (16, 26),
            (16, 31),
            (16, 33),
            (16, 40),
            (16, 42),
            (16, 48),
            (17, 0),
            (17, 2),
            (17, 3),
            (17, 10),
            (17, 18),
            (17, 21),
            (17, 26),
            (17, 31),
            (18, 0),
            (18, 2),
            (18, 3),
            (18, 10),
            (18, 18),
            (18, 21),
            (18, 26),
            (18, 31),
            (18, 48),
            (19, 0),
            (19, 2),
            (19, 3),
            (19, 10),
            (19, 18),
            (19, 19),
            (19, 21),
            (19, 26),
            (19, 31),
            (19, 33),
            (19, 40),
            (19, 42),
            (19, 48),
            (20, 0),
            (20, 2),
            (20, 10),
            (20, 26),
            (20, 31),
            (21, 0),
            (21, 2),
            (21, 3),
            (21, 10),
            (21, 18),
            (21, 21),
            (21, 26),
            (21, 31),
            (21, 33),
            (21, 40),
            (21, 48),
            (22, 0),
            (22, 2),
            (22, 3),
            (22, 10),
            (22, 18),
            (22, 21),
            (22, 26),
            (22, 31),
            (22, 48),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 10),
            (23, 14),
            (23, 15),
            (23, 18),
            (23, 19),
            (23, 21),
            (23, 23),
            (23, 24),
            (23, 26),
            (23, 31),
            (23, 33),
            (23, 34),
            (23, 38),
            (23, 40),
            (23, 41),
            (23, 42),
            (23, 43),
            (23, 44),
            (23, 45),
            (23, 48),
            (24, 0),
            (24, 10),
            (24, 26),
            (25, 0),
            (25, 10),
            (25, 26),
            (26, 0),
            (26, 2),
            (26, 3),
            (26, 10),
            (26, 18),
            (26, 21),
            (26, 26),
            (26, 31),
            (27, 0),
            (27, 10),
            (27, 26),
            (28, 0),
            (28, 2),
            (28, 10),
            (28, 18),
            (28, 26),
            (28, 31),
            (29, 0),
            (29, 2),
            (29, 3),
            (29, 10),
            (29, 18),
            (29, 21),
            (29, 26),
            (29, 31),
            (29, 48),
            (30, 0),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 6),
            (30, 10),
            (30, 14),
            (30, 18),
            (30, 19),
            (30, 21),
            (30, 23),
            (30, 24),
            (30, 26),
            (30, 31),
            (30, 33),
            (30, 40),
            (30, 42),
            (30, 44),
            (30, 45),
            (30, 48),
            (31, 0),
            (31, 2),
            (31, 3),
            (31, 10),
            (31, 18),
            (31, 21),
            (31, 26),
            (31, 31),
            (31, 33),
            (31, 40),
            (31, 42),
            (31, 48),
            (32, 0),
            (32, 10),
            (32, 26),
            (33, 0),
            (33, 2),
            (33, 3),
            (33, 10),
            (33, 18),
            (33, 21),
            (33, 26),
            (33, 31),
            (34, 0),
            (34, 2),
            (34, 3),
            (34, 10),
            (34, 18),
            (34, 21),
            (34, 26),
            (34, 31),
            (35, 0),
            (35, 2),
            (35, 3),
            (35, 10),
            (35, 18),
            (35, 21),
            (35, 26),
            (35, 31),
            (35, 33),
            (35, 40),
            (35, 48),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 10),
            (36, 14),
            (36, 15),
            (36, 16),
            (36, 18),
            (36, 19),
            (36, 21),
            (36, 23),
            (36, 24),
            (36, 26),
            (36, 29),
            (36, 30),
            (36, 31),
            (36, 32),
            (36, 33),
            (36, 34),
            (36, 38),
            (36, 40),
            (36, 41),
            (36, 42),
            (36, 43),
            (36, 44),
            (36, 45),
            (36, 48),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 6),
            (37, 10),
            (37, 14),
            (37, 18),
            (37, 19),
            (37, 21),
            (37, 23),
            (37, 24),
            (37, 26),
            (37, 31),
            (37, 33),
            (37, 34),
            (37, 38),
            (37, 40),
            (37, 42),
            (37, 43),
            (37, 44),
            (37, 48),
            (38, 0),
            (38, 10),
            (38, 26),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 10),
            (39, 12),
            (39, 14),
            (39, 15),
            (39, 16),
            (39, 18),
            (39, 19),
            (39, 20),
            (39, 21),
            (39, 22),
            (39, 23),
            (39, 24),
            (39, 25),
            (39, 26),
            (39, 27),
            (39, 29),
            (39, 30),
            (39, 31),
            (39, 32),
            (39, 33),
            (39, 34),
            (39, 35),
            (39, 38),
            (39, 40),
            (39, 41),
            (39, 42),
            (39, 43),
            (39, 44),
            (39, 45),
            (39, 48),
            (40, 26),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (41, 10),
            (41, 11),
            (41, 12),
            (41, 14),
            (41, 15),
            (41, 16),
            (41, 18),
            (41, 19),
            (41, 20),
            (41, 21),
            (41, 22),
            (41, 23),
            (41, 24),
            (41, 25),
            (41, 26),
            (41, 27),
            (41, 29),
            (41, 30),
            (41, 31),
            (41, 32),
            (41, 33),
            (41, 34),
            (41, 35),
            (41, 38),
            (41, 40),
            (41, 41),
            (41, 42),
            (41, 43),
            (41, 44),
            (41, 45),
            (41, 48),
            (41, 49),
            (42, 0),
            (42, 2),
            (42, 10),
            (42, 26),
            (42, 31),
            (43, 0),
            (43, 2),
            (43, 3),
            (43, 10),
            (43, 18),
            (43, 21),
            (43, 26),
            (43, 31),
            (44, 0),
            (44, 26),
            (45, 0),
            (45, 2),
            (45, 10),
            (45, 18),
            (45, 26),
            (45, 31),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 6),
            (46, 10),
            (46, 14),
            (46, 18),
            (46, 19),
            (46, 21),
            (46, 23),
            (46, 24),
            (46, 26),
            (46, 31),
            (46, 33),
            (46, 40),
            (46, 42),
            (46, 44),
            (46, 45),
            (46, 48),
            (47, 0),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 6),
            (47, 10),
            (47, 14),
            (47, 18),
            (47, 19),
            (47, 21),
            (47, 23),
            (47, 24),
            (47, 26),
            (47, 31),
            (47, 33),
            (47, 40),
            (47, 42),
            (47, 44),
            (47, 45),
            (47, 48),
            (48, 0),
            (48, 2),
            (48, 3),
            (48, 10),
            (48, 18),
            (48, 19),
            (48, 21),
            (48, 26),
            (48, 31),
            (48, 33),
            (48, 40),
            (48, 42),
            (48, 48),
            (49, 26),
        }:
            return 11
        elif key in {
            (0, 2),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 9),
            (0, 12),
            (0, 19),
            (0, 21),
            (0, 33),
            (0, 34),
            (0, 40),
            (0, 41),
            (0, 43),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (8, 33),
            (8, 40),
            (8, 45),
            (18, 33),
            (18, 40),
            (18, 45),
            (18, 46),
            (20, 33),
            (20, 40),
            (20, 45),
            (24, 33),
            (24, 40),
            (24, 45),
            (24, 46),
            (27, 33),
            (27, 40),
            (27, 45),
            (29, 45),
            (37, 45),
            (38, 33),
            (38, 40),
            (38, 45),
            (42, 4),
            (42, 6),
            (42, 12),
            (42, 33),
            (42, 40),
            (42, 41),
            (42, 43),
            (42, 45),
            (42, 46),
            (42, 48),
            (43, 6),
            (43, 33),
            (43, 40),
            (43, 43),
            (43, 45),
            (43, 46),
            (43, 48),
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
            (44, 15),
            (44, 17),
            (44, 18),
            (44, 19),
            (44, 20),
            (44, 21),
            (44, 22),
            (44, 23),
            (44, 24),
            (44, 25),
            (44, 27),
            (44, 28),
            (44, 29),
            (44, 30),
            (44, 31),
            (44, 32),
            (44, 33),
            (44, 34),
            (44, 35),
            (44, 36),
            (44, 37),
            (44, 38),
            (44, 39),
            (44, 40),
            (44, 41),
            (44, 42),
            (44, 43),
            (44, 45),
            (44, 46),
            (44, 47),
            (44, 48),
            (44, 49),
            (45, 33),
            (45, 40),
            (45, 45),
            (49, 45),
        }:
            return 33
        elif key in {(47, 35)}:
            return 1
        return 49

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, mlp_0_3_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_6_output, num_attn_1_0_output):
        key = (num_attn_1_6_output, num_attn_1_0_output)
        return 29

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_0_6_output):
        key = (num_attn_1_2_output, num_attn_0_6_output)
        return 41

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_6_output, num_attn_1_1_output):
        key = (num_attn_1_6_output, num_attn_1_1_output)
        return 28

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_7_output):
        key = num_attn_0_7_output
        return 25

    num_mlp_1_3_outputs = [num_mlp_1_3(k0) for k0 in num_attn_0_7_outputs]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 14, 31}:
            return position == 6
        elif mlp_0_0_output in {1, 43}:
            return position == 45
        elif mlp_0_0_output in {2}:
            return position == 12
        elif mlp_0_0_output in {3}:
            return position == 13
        elif mlp_0_0_output in {9, 4}:
            return position == 39
        elif mlp_0_0_output in {49, 5}:
            return position == 2
        elif mlp_0_0_output in {6, 7}:
            return position == 42
        elif mlp_0_0_output in {8, 26, 13}:
            return position == 0
        elif mlp_0_0_output in {48, 10}:
            return position == 8
        elif mlp_0_0_output in {11, 30}:
            return position == 15
        elif mlp_0_0_output in {40, 12}:
            return position == 4
        elif mlp_0_0_output in {15}:
            return position == 49
        elif mlp_0_0_output in {16}:
            return position == 5
        elif mlp_0_0_output in {17, 29}:
            return position == 7
        elif mlp_0_0_output in {41, 18, 38}:
            return position == 3
        elif mlp_0_0_output in {19}:
            return position == 28
        elif mlp_0_0_output in {27, 20}:
            return position == 21
        elif mlp_0_0_output in {21, 22, 23}:
            return position == 27
        elif mlp_0_0_output in {24}:
            return position == 26
        elif mlp_0_0_output in {25, 28}:
            return position == 37
        elif mlp_0_0_output in {32}:
            return position == 34
        elif mlp_0_0_output in {33}:
            return position == 41
        elif mlp_0_0_output in {34}:
            return position == 16
        elif mlp_0_0_output in {35}:
            return position == 35
        elif mlp_0_0_output in {36}:
            return position == 33
        elif mlp_0_0_output in {37}:
            return position == 1
        elif mlp_0_0_output in {39}:
            return position == 44
        elif mlp_0_0_output in {42}:
            return position == 38
        elif mlp_0_0_output in {44}:
            return position == 29
        elif mlp_0_0_output in {45}:
            return position == 30
        elif mlp_0_0_output in {46}:
            return position == 40
        elif mlp_0_0_output in {47}:
            return position == 43

    attn_2_0_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_5_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"0"}:
            return mlp_0_0_output == 41
        elif attn_0_1_output in {"1", "</s>"}:
            return mlp_0_0_output == 7
        elif attn_0_1_output in {"3", "4", "2"}:
            return mlp_0_0_output == 42
        elif attn_0_1_output in {"<s>"}:
            return mlp_0_0_output == 6

    attn_2_1_pattern = select_closest(mlp_0_0_outputs, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            2,
            3,
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
            29,
            30,
            31,
            32,
            33,
            35,
            36,
            38,
            39,
            40,
            42,
            43,
            44,
            45,
            46,
            48,
            49,
        }:
            return token == ""
        elif mlp_0_0_output in {1}:
            return token == "</s>"
        elif mlp_0_0_output in {9, 18}:
            return token == "2"
        elif mlp_0_0_output in {34, 37}:
            return token == "0"
        elif mlp_0_0_output in {41}:
            return token == "1"
        elif mlp_0_0_output in {47}:
            return token == "<s>"

    attn_2_2_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_1_6_output, token):
        if attn_1_6_output in {"0", "</s>"}:
            return token == "<s>"
        elif attn_1_6_output in {"1"}:
            return token == ""
        elif attn_1_6_output in {"2"}:
            return token == "3"
        elif attn_1_6_output in {"3"}:
            return token == "4"
        elif attn_1_6_output in {"4"}:
            return token == "2"
        elif attn_1_6_output in {"<s>"}:
            return token == "</s>"

    attn_2_3_pattern = select_closest(tokens, attn_1_6_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_7_output, token):
        if attn_0_7_output in {"1", "4", "0", "3", "2"}:
            return token == ""
        elif attn_0_7_output in {"</s>"}:
            return token == "<s>"
        elif attn_0_7_output in {"<s>"}:
            return token == "</s>"

    attn_2_4_pattern = select_closest(tokens, attn_0_7_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_4_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_2_output, token):
        if attn_0_2_output in {"0", "</s>"}:
            return token == "<s>"
        elif attn_0_2_output in {"1", "<s>"}:
            return token == "</s>"
        elif attn_0_2_output in {"3", "4", "2"}:
            return token == ""

    attn_2_5_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_3_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_1_6_output, token):
        if attn_1_6_output in {"0", "1", "</s>"}:
            return token == "<s>"
        elif attn_1_6_output in {"4", "2"}:
            return token == ""
        elif attn_1_6_output in {"<s>", "3"}:
            return token == "</s>"

    attn_2_6_pattern = select_closest(tokens, attn_1_6_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_0_6_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 42}:
            return k_mlp_0_0_output == 42
        elif q_mlp_0_0_output in {1}:
            return k_mlp_0_0_output == 11
        elif q_mlp_0_0_output in {2, 7}:
            return k_mlp_0_0_output == 7
        elif q_mlp_0_0_output in {17, 3}:
            return k_mlp_0_0_output == 9
        elif q_mlp_0_0_output in {4, 21}:
            return k_mlp_0_0_output == 28
        elif q_mlp_0_0_output in {13, 5}:
            return k_mlp_0_0_output == 48
        elif q_mlp_0_0_output in {6}:
            return k_mlp_0_0_output == 13
        elif q_mlp_0_0_output in {8}:
            return k_mlp_0_0_output == 19
        elif q_mlp_0_0_output in {9, 15}:
            return k_mlp_0_0_output == 17
        elif q_mlp_0_0_output in {10}:
            return k_mlp_0_0_output == 34
        elif q_mlp_0_0_output in {11}:
            return k_mlp_0_0_output == 45
        elif q_mlp_0_0_output in {24, 12, 29}:
            return k_mlp_0_0_output == 29
        elif q_mlp_0_0_output in {14}:
            return k_mlp_0_0_output == 5
        elif q_mlp_0_0_output in {16, 37}:
            return k_mlp_0_0_output == 37
        elif q_mlp_0_0_output in {18, 46}:
            return k_mlp_0_0_output == 18
        elif q_mlp_0_0_output in {19}:
            return k_mlp_0_0_output == 12
        elif q_mlp_0_0_output in {33, 20}:
            return k_mlp_0_0_output == 31
        elif q_mlp_0_0_output in {49, 22, 30}:
            return k_mlp_0_0_output == 6
        elif q_mlp_0_0_output in {43, 27, 23}:
            return k_mlp_0_0_output == 25
        elif q_mlp_0_0_output in {25}:
            return k_mlp_0_0_output == 21
        elif q_mlp_0_0_output in {26, 47}:
            return k_mlp_0_0_output == 16
        elif q_mlp_0_0_output in {28}:
            return k_mlp_0_0_output == 1
        elif q_mlp_0_0_output in {40, 31}:
            return k_mlp_0_0_output == 44
        elif q_mlp_0_0_output in {32}:
            return k_mlp_0_0_output == 33
        elif q_mlp_0_0_output in {34}:
            return k_mlp_0_0_output == 36
        elif q_mlp_0_0_output in {35}:
            return k_mlp_0_0_output == 3
        elif q_mlp_0_0_output in {36}:
            return k_mlp_0_0_output == 47
        elif q_mlp_0_0_output in {45, 38}:
            return k_mlp_0_0_output == 30
        elif q_mlp_0_0_output in {39}:
            return k_mlp_0_0_output == 49
        elif q_mlp_0_0_output in {41}:
            return k_mlp_0_0_output == 40
        elif q_mlp_0_0_output in {44}:
            return k_mlp_0_0_output == 35
        elif q_mlp_0_0_output in {48}:
            return k_mlp_0_0_output == 26

    attn_2_7_pattern = select_closest(mlp_0_0_outputs, mlp_0_0_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_3_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_1_0_output):
        if attn_1_2_output in {"0", "2"}:
            return attn_1_0_output == "</s>"
        elif attn_1_2_output in {"1"}:
            return attn_1_0_output == ""
        elif attn_1_2_output in {"3"}:
            return attn_1_0_output == "<s>"
        elif attn_1_2_output in {"</s>", "<s>", "4"}:
            return attn_1_0_output == "2"

    num_attn_2_0_pattern = select(attn_1_0_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_6_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_1_7_output):
        if position in {0, 2, 3, 6, 40, 12, 45}:
            return attn_1_7_output == "1"
        elif position in {1, 9, 14}:
            return attn_1_7_output == "</s>"
        elif position in {
            4,
            10,
            13,
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
            42,
            43,
            47,
            48,
            49,
        }:
            return attn_1_7_output == ""
        elif position in {11, 5}:
            return attn_1_7_output == "<s>"
        elif position in {8, 44, 7}:
            return attn_1_7_output == "0"
        elif position in {46, 15}:
            return attn_1_7_output == "<pad>"

    num_attn_2_1_pattern = select(attn_1_7_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_2_output, attn_1_0_output):
        if attn_1_2_output in {"0", "1", "<s>", "</s>"}:
            return attn_1_0_output == "0"
        elif attn_1_2_output in {"2"}:
            return attn_1_0_output == "<s>"
        elif attn_1_2_output in {"4", "3"}:
            return attn_1_0_output == ""

    num_attn_2_2_pattern = select(attn_1_0_outputs, attn_1_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_2_output, attn_0_6_output):
        if attn_1_2_output in {"0"}:
            return attn_0_6_output == "<s>"
        elif attn_1_2_output in {"1"}:
            return attn_0_6_output == "2"
        elif attn_1_2_output in {"2"}:
            return attn_0_6_output == "1"
        elif attn_1_2_output in {"</s>", "4", "<s>", "3"}:
            return attn_0_6_output == ""

    num_attn_2_3_pattern = select(attn_0_6_outputs, attn_1_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_5_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_2_output, position):
        if attn_1_2_output in {"0"}:
            return position == 8
        elif attn_1_2_output in {"1"}:
            return position == 1
        elif attn_1_2_output in {"2"}:
            return position == 30
        elif attn_1_2_output in {"3"}:
            return position == 26
        elif attn_1_2_output in {"4"}:
            return position == 28
        elif attn_1_2_output in {"</s>"}:
            return position == 10
        elif attn_1_2_output in {"<s>"}:
            return position == 21

    num_attn_2_4_pattern = select(positions, attn_1_2_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_6_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_2_output, position):
        if attn_1_2_output in {"0"}:
            return position == 33
        elif attn_1_2_output in {"1", "2"}:
            return position == 35
        elif attn_1_2_output in {"3"}:
            return position == 1
        elif attn_1_2_output in {"4"}:
            return position == 25
        elif attn_1_2_output in {"</s>"}:
            return position == 28
        elif attn_1_2_output in {"<s>"}:
            return position == 43

    num_attn_2_5_pattern = select(positions, attn_1_2_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_6_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_2_output, attn_1_7_output):
        if attn_1_2_output in {"0", "<s>", "4"}:
            return attn_1_7_output == ""
        elif attn_1_2_output in {"1", "</s>"}:
            return attn_1_7_output == "1"
        elif attn_1_2_output in {"2"}:
            return attn_1_7_output == "</s>"
        elif attn_1_2_output in {"3"}:
            return attn_1_7_output == "<s>"

    num_attn_2_6_pattern = select(attn_1_7_outputs, attn_1_2_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_6_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_2_output, attn_1_0_output):
        if attn_1_2_output in {"0", "<s>"}:
            return attn_1_0_output == "1"
        elif attn_1_2_output in {"1", "4", "</s>", "3", "2"}:
            return attn_1_0_output == ""

    num_attn_2_7_pattern = select(attn_1_0_outputs, attn_1_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_6_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_0_output, mlp_1_1_output):
        key = (num_mlp_0_0_output, mlp_1_1_output)
        if key in {
            (2, 26),
            (17, 15),
            (17, 26),
            (17, 35),
            (23, 0),
            (23, 1),
            (23, 3),
            (23, 4),
            (23, 6),
            (23, 9),
            (23, 10),
            (23, 11),
            (23, 12),
            (23, 13),
            (23, 14),
            (23, 15),
            (23, 18),
            (23, 19),
            (23, 20),
            (23, 21),
            (23, 22),
            (23, 23),
            (23, 24),
            (23, 26),
            (23, 33),
            (23, 34),
            (23, 35),
            (23, 38),
            (23, 40),
            (23, 41),
            (23, 42),
            (23, 44),
            (23, 45),
            (23, 46),
            (23, 47),
            (23, 48),
            (23, 49),
            (43, 15),
            (43, 26),
            (43, 35),
        }:
            return 9
        return 14

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, mlp_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_6_output, num_mlp_1_1_output):
        key = (attn_2_6_output, num_mlp_1_1_output)
        return 17

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_6_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_5_output, attn_0_1_output):
        key = (attn_2_5_output, attn_0_1_output)
        if key in {("1", "0")}:
            return 38
        return 14

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_5_outputs, attn_0_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(mlp_0_1_output, num_mlp_0_2_output):
        key = (mlp_0_1_output, num_mlp_0_2_output)
        return 1

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, num_mlp_0_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_4_output, num_attn_1_6_output):
        key = (num_attn_2_4_output, num_attn_1_6_output)
        return 22

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_4_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_7_output):
        key = num_attn_1_7_output
        return 36

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_1_7_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_6_output):
        key = num_attn_1_6_output
        return 21

    num_mlp_2_2_outputs = [num_mlp_2_2(k0) for k0 in num_attn_1_6_outputs]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_1_output, num_attn_1_1_output):
        key = (num_attn_2_1_output, num_attn_1_1_output)
        return 25

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_1_outputs)
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


print(
    run(
        [
            "<s>",
            "0",
            "3",
            "2",
            "3",
            "0",
            "2",
            "1",
            "3",
            "2",
            "4",
            "4",
            "4",
            "3",
            "4",
            "2",
            "3",
            "</s>",
        ]
    )
)
