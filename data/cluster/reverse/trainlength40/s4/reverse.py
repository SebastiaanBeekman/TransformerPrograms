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
        "output/length/rasp/reverse/trainlength40/s4/reverse_weights.csv",
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
        if q_position in {0, 35, 36, 37, 38}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 21
        elif q_position in {2}:
            return k_position == 17
        elif q_position in {3}:
            return k_position == 34
        elif q_position in {4}:
            return k_position == 35
        elif q_position in {5}:
            return k_position == 32
        elif q_position in {6}:
            return k_position == 29
        elif q_position in {7}:
            return k_position == 23
        elif q_position in {8}:
            return k_position == 31
        elif q_position in {9}:
            return k_position == 27
        elif q_position in {49, 10}:
            return k_position == 25
        elif q_position in {11}:
            return k_position == 33
        elif q_position in {12, 13}:
            return k_position == 22
        elif q_position in {14, 15, 16, 19, 20, 22, 23, 24, 25}:
            return k_position == 0
        elif q_position in {17, 29}:
            return k_position == 8
        elif q_position in {18, 30}:
            return k_position == 5
        elif q_position in {21}:
            return k_position == 39
        elif q_position in {26}:
            return k_position == 10
        elif q_position in {43, 27}:
            return k_position == 4
        elif q_position in {28}:
            return k_position == 6
        elif q_position in {34, 31}:
            return k_position == 2
        elif q_position in {32, 33, 47}:
            return k_position == 3
        elif q_position in {39}:
            return k_position == 16
        elif q_position in {40}:
            return k_position == 18
        elif q_position in {41}:
            return k_position == 38
        elif q_position in {42, 45}:
            return k_position == 19
        elif q_position in {44}:
            return k_position == 24
        elif q_position in {46}:
            return k_position == 37
        elif q_position in {48}:
            return k_position == 48

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 38, 39}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {10, 3, 42, 13}:
            return k_position == 26
        elif q_position in {4}:
            return k_position == 19
        elif q_position in {21, 5}:
            return k_position == 15
        elif q_position in {6, 15}:
            return k_position == 23
        elif q_position in {7}:
            return k_position == 21
        elif q_position in {8}:
            return k_position == 20
        elif q_position in {9}:
            return k_position == 30
        elif q_position in {16, 18, 11}:
            return k_position == 0
        elif q_position in {12}:
            return k_position == 27
        elif q_position in {32, 33, 40, 14}:
            return k_position == 5
        elif q_position in {17, 34}:
            return k_position == 4
        elif q_position in {26, 19, 28}:
            return k_position == 9
        elif q_position in {20, 29, 30}:
            return k_position == 7
        elif q_position in {25, 27, 22}:
            return k_position == 11
        elif q_position in {31, 23}:
            return k_position == 6
        elif q_position in {24}:
            return k_position == 14
        elif q_position in {35, 36, 37}:
            return k_position == 2
        elif q_position in {41, 44}:
            return k_position == 49
        elif q_position in {43}:
            return k_position == 28
        elif q_position in {45}:
            return k_position == 24
        elif q_position in {46, 47}:
            return k_position == 38
        elif q_position in {48}:
            return k_position == 3
        elif q_position in {49}:
            return k_position == 39

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 8, 36, 15}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 22
        elif q_position in {40, 49, 2}:
            return k_position == 31
        elif q_position in {3, 23}:
            return k_position == 36
        elif q_position in {9, 4}:
            return k_position == 24
        elif q_position in {11, 5, 7}:
            return k_position == 26
        elif q_position in {6}:
            return k_position == 33
        elif q_position in {41, 10}:
            return k_position == 29
        elif q_position in {12}:
            return k_position == 17
        elif q_position in {13, 14}:
            return k_position == 23
        elif q_position in {16, 43}:
            return k_position == 4
        elif q_position in {17, 29}:
            return k_position == 6
        elif q_position in {18, 22, 47}:
            return k_position == 39
        elif q_position in {19}:
            return k_position == 0
        elif q_position in {20, 31}:
            return k_position == 8
        elif q_position in {32, 21}:
            return k_position == 5
        elif q_position in {24, 25}:
            return k_position == 13
        elif q_position in {26, 35, 34}:
            return k_position == 3
        elif q_position in {27, 30}:
            return k_position == 9
        elif q_position in {28}:
            return k_position == 10
        elif q_position in {33, 37, 38, 39}:
            return k_position == 1
        elif q_position in {42}:
            return k_position == 46
        elif q_position in {44}:
            return k_position == 48
        elif q_position in {45}:
            return k_position == 40
        elif q_position in {46}:
            return k_position == 49
        elif q_position in {48}:
            return k_position == 25

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 38, 39}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 14
        elif q_position in {9, 10, 3, 47}:
            return k_position == 28
        elif q_position in {11, 4}:
            return k_position == 17
        elif q_position in {5}:
            return k_position == 24
        elif q_position in {48, 6, 7}:
            return k_position == 32
        elif q_position in {8}:
            return k_position == 29
        elif q_position in {12}:
            return k_position == 18
        elif q_position in {42, 13, 14}:
            return k_position == 20
        elif q_position in {16, 17, 15}:
            return k_position == 22
        elif q_position in {33, 35, 18, 19, 20, 26}:
            return k_position == 4
        elif q_position in {25, 37, 21}:
            return k_position == 2
        elif q_position in {24, 28, 22, 36}:
            return k_position == 3
        elif q_position in {46, 23}:
            return k_position == 39
        elif q_position in {27}:
            return k_position == 6
        elif q_position in {34, 29, 30, 31}:
            return k_position == 5
        elif q_position in {32}:
            return k_position == 7
        elif q_position in {40}:
            return k_position == 38
        elif q_position in {41}:
            return k_position == 36
        elif q_position in {43}:
            return k_position == 15
        elif q_position in {44}:
            return k_position == 37
        elif q_position in {45}:
            return k_position == 44
        elif q_position in {49}:
            return k_position == 42

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 36, 37, 38, 39, 28, 31}:
            return k_position == 1
        elif q_position in {48, 1, 3}:
            return k_position == 35
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {27, 4, 30}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 34
        elif q_position in {8, 6}:
            return k_position == 28
        elif q_position in {7}:
            return k_position == 30
        elif q_position in {9, 43}:
            return k_position == 12
        elif q_position in {10, 12}:
            return k_position == 16
        elif q_position in {11, 44, 13, 14, 15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 21
        elif q_position in {17, 49, 23}:
            return k_position == 5
        elif q_position in {18, 35, 20}:
            return k_position == 3
        elif q_position in {32, 33, 19, 24, 25, 26}:
            return k_position == 6
        elif q_position in {29, 34, 21, 22}:
            return k_position == 4
        elif q_position in {40}:
            return k_position == 15
        elif q_position in {41}:
            return k_position == 49
        elif q_position in {42}:
            return k_position == 11
        elif q_position in {45}:
            return k_position == 14
        elif q_position in {46, 47}:
            return k_position == 45

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_token, k_token):
        if q_token in {"<s>", "0"}:
            return k_token == "4"
        elif q_token in {"2", "3", "</s>", "1", "4"}:
            return k_token == "0"

    attn_0_5_pattern = select_closest(tokens, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 27, 38, 39}:
            return k_position == 1
        elif q_position in {1, 22}:
            return k_position == 7
        elif q_position in {2, 21, 30}:
            return k_position == 6
        elif q_position in {3, 45}:
            return k_position == 27
        elif q_position in {49, 4, 12, 46}:
            return k_position == 23
        elif q_position in {5}:
            return k_position == 30
        elif q_position in {44, 29, 6}:
            return k_position == 10
        elif q_position in {34, 35, 36, 37, 7, 10}:
            return k_position == 2
        elif q_position in {8}:
            return k_position == 24
        elif q_position in {9}:
            return k_position == 26
        elif q_position in {11}:
            return k_position == 28
        elif q_position in {13, 14}:
            return k_position == 25
        elif q_position in {48, 42, 15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 21
        elif q_position in {41, 18}:
            return k_position == 20
        elif q_position in {19, 23}:
            return k_position == 3
        elif q_position in {20}:
            return k_position == 9
        elif q_position in {24, 25, 33}:
            return k_position == 5
        elif q_position in {26}:
            return k_position == 12
        elif q_position in {28}:
            return k_position == 8
        elif q_position in {32, 43, 31}:
            return k_position == 4
        elif q_position in {40}:
            return k_position == 40
        elif q_position in {47}:
            return k_position == 15

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 38, 32, 30}:
            return k_position == 1
        elif q_position in {16, 1, 17}:
            return k_position == 20
        elif q_position in {2}:
            return k_position == 15
        elif q_position in {48, 3, 5}:
            return k_position == 29
        elif q_position in {4}:
            return k_position == 33
        elif q_position in {10, 6}:
            return k_position == 27
        elif q_position in {7}:
            return k_position == 30
        elif q_position in {8}:
            return k_position == 26
        elif q_position in {9}:
            return k_position == 25
        elif q_position in {34, 36, 11, 15, 21, 25, 27}:
            return k_position == 3
        elif q_position in {35, 37, 39, 12, 14, 18, 19, 20, 23, 24, 26, 28, 29, 31}:
            return k_position == 2
        elif q_position in {13}:
            return k_position == 17
        elif q_position in {22}:
            return k_position == 39
        elif q_position in {33, 40, 42, 43, 44, 49}:
            return k_position == 4
        elif q_position in {41}:
            return k_position == 31
        elif q_position in {45, 47}:
            return k_position == 47
        elif q_position in {46}:
            return k_position == 46

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 41
        elif q_position in {1, 2, 40, 9, 30}:
            return k_position == 43
        elif q_position in {34, 3, 35, 11, 12, 14, 46}:
            return k_position == 45
        elif q_position in {4}:
            return k_position == 8
        elif q_position in {5, 23}:
            return k_position == 46
        elif q_position in {37, 6, 13, 17, 21}:
            return k_position == 40
        elif q_position in {7}:
            return k_position == 14
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16, 42}:
            return k_position == 48
        elif q_position in {36, 18, 19, 20, 27}:
            return k_position == 44
        elif q_position in {24, 41, 43, 22}:
            return k_position == 49
        elif q_position in {25}:
            return k_position == 37
        elif q_position in {39, 45, 49, 26, 29}:
            return k_position == 47
        elif q_position in {28}:
            return k_position == 34
        elif q_position in {31}:
            return k_position == 35
        elif q_position in {32, 47}:
            return k_position == 42
        elif q_position in {33}:
            return k_position == 36
        elif q_position in {44, 38}:
            return k_position == 39
        elif q_position in {48}:
            return k_position == 0

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 9, 11, 16, 21}:
            return k_position == 40
        elif q_position in {1, 2, 4, 12, 18, 19}:
            return k_position == 45
        elif q_position in {3}:
            return k_position == 18
        elif q_position in {43, 5, 14}:
            return k_position == 43
        elif q_position in {33, 37, 6, 13, 20, 31}:
            return k_position == 49
        elif q_position in {7}:
            return k_position == 33
        elif q_position in {8, 40, 48, 39}:
            return k_position == 44
        elif q_position in {25, 10, 42, 44}:
            return k_position == 48
        elif q_position in {32, 35, 15, 17, 28}:
            return k_position == 46
        elif q_position in {24, 27, 22, 47}:
            return k_position == 41
        elif q_position in {23}:
            return k_position == 29
        elif q_position in {41, 26}:
            return k_position == 42
        elif q_position in {36, 29}:
            return k_position == 47
        elif q_position in {30}:
            return k_position == 31
        elif q_position in {34}:
            return k_position == 1
        elif q_position in {38}:
            return k_position == 32
        elif q_position in {45, 46}:
            return k_position == 38
        elif q_position in {49}:
            return k_position == 36

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {
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
            13,
            15,
            16,
            18,
            19,
            21,
            23,
            24,
            25,
            26,
            28,
            29,
            30,
            32,
            33,
            34,
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
            return token == ""
        elif position in {27, 12, 31}:
            return token == "</s>"
        elif position in {35, 36, 37, 14, 17, 20, 22}:
            return token == "<s>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"0"}:
            return position == 47
        elif token in {"1"}:
            return position == 49
        elif token in {"<s>", "2"}:
            return position == 41
        elif token in {"3"}:
            return position == 45
        elif token in {"4"}:
            return position == 40
        elif token in {"</s>"}:
            return position == 11

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {
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
            40,
            41,
            42,
            43,
            44,
            45,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {46}:
            return token == "<pad>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_token, k_token):
        if q_token in {"<s>", "2", "3", "0", "1", "4"}:
            return k_token == ""
        elif q_token in {"</s>"}:
            return k_token == "<s>"

    num_attn_0_5_pattern = select(tokens, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0, 31}:
            return k_position == 47
        elif q_position in {1, 35}:
            return k_position == 48
        elif q_position in {33, 2, 36, 6, 7, 40, 28}:
            return k_position == 39
        elif q_position in {42, 3, 37, 15}:
            return k_position == 46
        elif q_position in {4, 21, 23}:
            return k_position == 38
        elif q_position in {32, 5, 9, 10, 44, 47, 16, 48}:
            return k_position == 37
        elif q_position in {8, 49}:
            return k_position == 35
        elif q_position in {18, 11, 30}:
            return k_position == 34
        elif q_position in {27, 12, 38}:
            return k_position == 49
        elif q_position in {43, 13, 22}:
            return k_position == 41
        elif q_position in {34, 14, 39}:
            return k_position == 42
        elif q_position in {17}:
            return k_position == 45
        elif q_position in {19}:
            return k_position == 43
        elif q_position in {20}:
            return k_position == 31
        elif q_position in {24}:
            return k_position == 30
        elif q_position in {25}:
            return k_position == 32
        elif q_position in {26, 45}:
            return k_position == 36
        elif q_position in {29}:
            return k_position == 33
        elif q_position in {41}:
            return k_position == 40
        elif q_position in {46}:
            return k_position == 11

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"0"}:
            return position == 44
        elif token in {"3", "1"}:
            return position == 49
        elif token in {"2"}:
            return position == 43
        elif token in {"<s>", "4"}:
            return position == 40
        elif token in {"</s>"}:
            return position == 20

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output):
        key = attn_0_0_output
        if key in {"", "0", "1", "2", "<pad>"}:
            return 2
        return 3

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_0_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_4_output):
        key = (attn_0_2_output, attn_0_4_output)
        return 2

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_4_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output):
        key = num_attn_0_5_output
        return 4

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_5_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        return 5

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 1, 2}:
            return token == "<s>"
        elif position in {32, 34, 3, 4, 5, 36, 28, 30}:
            return token == "</s>"
        elif position in {
            6,
            7,
            9,
            11,
            12,
            13,
            14,
            24,
            27,
            29,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            48,
            49,
        }:
            return token == "2"
        elif position in {8, 18, 19, 20, 25}:
            return token == "0"
        elif position in {33, 37, 38, 39, 10, 47, 16}:
            return token == "3"
        elif position in {35, 15, 17, 21, 31}:
            return token == "4"
        elif position in {22, 23}:
            return token == "1"
        elif position in {26}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_7_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_2_output, attn_0_0_output):
        if attn_0_2_output in {"0", "</s>"}:
            return attn_0_0_output == ""
        elif attn_0_2_output in {"<s>", "2", "3", "1", "4"}:
            return attn_0_0_output == "<s>"

    attn_1_1_pattern = select_closest(attn_0_0_outputs, attn_0_2_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 4, 5, 8, 9, 42, 44, 14, 18, 23, 31}:
            return token == ""
        elif mlp_0_0_output in {1, 2, 3, 7, 13, 15, 28}:
            return token == "3"
        elif mlp_0_0_output in {38, 6}:
            return token == "0"
        elif mlp_0_0_output in {35, 41, 10, 12, 16}:
            return token == "4"
        elif mlp_0_0_output in {
            33,
            34,
            39,
            40,
            11,
            46,
            47,
            48,
            17,
            49,
            20,
            24,
            26,
            27,
            29,
        }:
            return token == "2"
        elif mlp_0_0_output in {32, 36, 37, 43, 45, 19, 21, 22, 25, 30}:
            return token == "1"

    attn_1_2_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, attn_0_6_output):
        if attn_0_3_output in {"0", "</s>"}:
            return attn_0_6_output == "</s>"
        elif attn_0_3_output in {"1", "2"}:
            return attn_0_6_output == "0"
        elif attn_0_3_output in {"3"}:
            return attn_0_6_output == "1"
        elif attn_0_3_output in {"4"}:
            return attn_0_6_output == "2"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_6_output == "<s>"

    attn_1_3_pattern = select_closest(attn_0_6_outputs, attn_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_7_output, token):
        if attn_0_7_output in {"3", "0", "1", "2"}:
            return token == ""
        elif attn_0_7_output in {"4"}:
            return token == "<s>"
        elif attn_0_7_output in {"</s>"}:
            return token == "</s>"
        elif attn_0_7_output in {"<s>"}:
            return token == "1"

    attn_1_4_pattern = select_closest(tokens, attn_0_7_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_3_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 8
        elif q_position in {1, 2}:
            return k_position == 3
        elif q_position in {42, 3, 45, 47}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 16
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16, 21}:
            return k_position == 23
        elif q_position in {17}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 20
        elif q_position in {19}:
            return k_position == 21
        elif q_position in {20}:
            return k_position == 22
        elif q_position in {22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 25
        elif q_position in {24, 25}:
            return k_position == 27
        elif q_position in {43, 26, 27, 28}:
            return k_position == 30
        elif q_position in {44, 29, 30}:
            return k_position == 32
        elif q_position in {31}:
            return k_position == 33
        elif q_position in {32, 33}:
            return k_position == 34
        elif q_position in {34, 36}:
            return k_position == 35
        elif q_position in {35, 37}:
            return k_position == 36
        elif q_position in {38}:
            return k_position == 37
        elif q_position in {39}:
            return k_position == 38
        elif q_position in {40, 41}:
            return k_position == 44
        elif q_position in {46}:
            return k_position == 39
        elif q_position in {48}:
            return k_position == 47
        elif q_position in {49}:
            return k_position == 29

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_0_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_5_output, attn_0_1_output):
        if attn_0_5_output in {"0", "4"}:
            return attn_0_1_output == "<s>"
        elif attn_0_5_output in {"1"}:
            return attn_0_1_output == "<pad>"
        elif attn_0_5_output in {"3", "</s>", "<s>", "2"}:
            return attn_0_1_output == ""

    attn_1_6_pattern = select_closest(attn_0_1_outputs, attn_0_5_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_6_output, token):
        if attn_0_6_output in {"0", "1"}:
            return token == "3"
        elif attn_0_6_output in {"</s>", "2"}:
            return token == "2"
        elif attn_0_6_output in {"3"}:
            return token == "4"
        elif attn_0_6_output in {"4"}:
            return token == "0"
        elif attn_0_6_output in {"<s>"}:
            return token == "<s>"

    attn_1_7_pattern = select_closest(tokens, attn_0_6_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_5_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_4_output, attn_0_3_output):
        if attn_0_4_output in {"2", "3", "</s>", "0", "1", "4"}:
            return attn_0_3_output == ""
        elif attn_0_4_output in {"<s>"}:
            return attn_0_3_output == "3"

    num_attn_1_0_pattern = select(attn_0_3_outputs, attn_0_4_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_7_output, attn_0_0_output):
        if attn_0_7_output in {"<s>", "2", "</s>", "0", "1", "4"}:
            return attn_0_0_output == ""
        elif attn_0_7_output in {"3"}:
            return attn_0_0_output == "3"

    num_attn_1_1_pattern = select(attn_0_0_outputs, attn_0_7_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_7_output, attn_0_3_output):
        if attn_0_7_output in {"2", "3", "0", "1", "4"}:
            return attn_0_3_output == ""
        elif attn_0_7_output in {"<s>", "</s>"}:
            return attn_0_3_output == "2"

    num_attn_1_2_pattern = select(attn_0_3_outputs, attn_0_7_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_attn_0_4_output, k_attn_0_4_output):
        if q_attn_0_4_output in {"<s>", "3", "</s>", "0", "1", "4"}:
            return k_attn_0_4_output == ""
        elif q_attn_0_4_output in {"2"}:
            return k_attn_0_4_output == "2"

    num_attn_1_3_pattern = select(attn_0_4_outputs, attn_0_4_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {
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
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
            30,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            42,
            43,
            44,
            45,
            46,
            47,
            49,
        }:
            return token == ""
        elif num_mlp_0_0_output in {25, 11, 21}:
            return token == "1"
        elif num_mlp_0_0_output in {20}:
            return token == "<s>"
        elif num_mlp_0_0_output in {31}:
            return token == "</s>"
        elif num_mlp_0_0_output in {41}:
            return token == "3"
        elif num_mlp_0_0_output in {48}:
            return token == "<pad>"

    num_attn_1_4_pattern = select(tokens, num_mlp_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_3_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_5_output, attn_0_1_output):
        if attn_0_5_output in {"<s>", "2", "3", "</s>", "0", "1", "4"}:
            return attn_0_1_output == ""

    num_attn_1_5_pattern = select(attn_0_1_outputs, attn_0_5_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_6_output, attn_0_3_output):
        if attn_0_6_output in {"<s>", "2", "3", "0", "1", "4"}:
            return attn_0_3_output == ""
        elif attn_0_6_output in {"</s>"}:
            return attn_0_3_output == "0"

    num_attn_1_6_pattern = select(attn_0_3_outputs, attn_0_6_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_3_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_4_output, attn_0_1_output):
        if attn_0_4_output in {"<s>", "2", "3", "</s>", "0", "1", "4"}:
            return attn_0_1_output == ""

    num_attn_1_7_pattern = select(attn_0_1_outputs, attn_0_4_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_6_output, attn_1_5_output):
        key = (attn_0_6_output, attn_1_5_output)
        return 12

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_1_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_2_output, attn_0_0_output):
        key = (attn_1_2_output, attn_0_0_output)
        return 3

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output, num_attn_1_0_output):
        key = (num_attn_1_7_output, num_attn_1_0_output)
        return 29

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_5_output, num_attn_1_2_output):
        key = (num_attn_1_5_output, num_attn_1_2_output)
        return 5

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_4_output, attn_1_6_output):
        if attn_0_4_output in {"0"}:
            return attn_1_6_output == "</s>"
        elif attn_0_4_output in {"1", "4"}:
            return attn_1_6_output == "3"
        elif attn_0_4_output in {"3", "</s>", "<s>", "2"}:
            return attn_1_6_output == ""

    attn_2_0_pattern = select_closest(attn_1_6_outputs, attn_0_4_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, token):
        if attn_0_1_output in {"0", "</s>"}:
            return token == "3"
        elif attn_0_1_output in {"1"}:
            return token == "<s>"
        elif attn_0_1_output in {"3", "2"}:
            return token == ""
        elif attn_0_1_output in {"<s>", "4"}:
            return token == "</s>"

    attn_2_1_pattern = select_closest(tokens, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_attn_0_7_output, k_attn_0_7_output):
        if q_attn_0_7_output in {"<s>", "</s>", "0", "1", "4"}:
            return k_attn_0_7_output == ""
        elif q_attn_0_7_output in {"2"}:
            return k_attn_0_7_output == "</s>"
        elif q_attn_0_7_output in {"3"}:
            return k_attn_0_7_output == "<s>"

    attn_2_2_pattern = select_closest(attn_0_7_outputs, attn_0_7_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_7_output, position):
        if attn_0_7_output in {"0"}:
            return position == 44
        elif attn_0_7_output in {"1"}:
            return position == 20
        elif attn_0_7_output in {"2"}:
            return position == 12
        elif attn_0_7_output in {"3"}:
            return position == 13
        elif attn_0_7_output in {"<s>", "4"}:
            return position == 4
        elif attn_0_7_output in {"</s>"}:
            return position == 9

    attn_2_3_pattern = select_closest(positions, attn_0_7_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_4_output, token):
        if attn_0_4_output in {"0", "2"}:
            return token == "4"
        elif attn_0_4_output in {"1", "4"}:
            return token == "0"
        elif attn_0_4_output in {"3"}:
            return token == "2"
        elif attn_0_4_output in {"</s>"}:
            return token == "3"
        elif attn_0_4_output in {"<s>"}:
            return token == ""

    attn_2_4_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_1_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_5_output, token):
        if attn_0_5_output in {"<s>", "0", "4"}:
            return token == "3"
        elif attn_0_5_output in {"3", "1", "</s>"}:
            return token == ""
        elif attn_0_5_output in {"2"}:
            return token == "2"

    attn_2_5_pattern = select_closest(tokens, attn_0_5_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_6_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "2"
        elif attn_0_2_output in {"1"}:
            return token == ""
        elif attn_0_2_output in {"3", "4", "2"}:
            return token == "1"
        elif attn_0_2_output in {"</s>"}:
            return token == "4"
        elif attn_0_2_output in {"<s>"}:
            return token == "</s>"

    attn_2_6_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_6_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_7_output, mlp_1_1_output):
        if attn_0_7_output in {"0"}:
            return mlp_1_1_output == 42
        elif attn_0_7_output in {"1"}:
            return mlp_1_1_output == 18
        elif attn_0_7_output in {"<s>", "4", "2"}:
            return mlp_1_1_output == 24
        elif attn_0_7_output in {"3"}:
            return mlp_1_1_output == 0
        elif attn_0_7_output in {"</s>"}:
            return mlp_1_1_output == 19

    attn_2_7_pattern = select_closest(mlp_1_1_outputs, attn_0_7_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_6_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_0_2_output):
        if attn_1_2_output in {"0"}:
            return attn_0_2_output == "0"
        elif attn_1_2_output in {"<s>", "2", "3", "</s>", "1", "4"}:
            return attn_0_2_output == ""

    num_attn_2_0_pattern = select(attn_0_2_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_attn_1_5_output, k_attn_1_5_output):
        if q_attn_1_5_output in {"<s>", "3", "</s>", "0", "1", "2"}:
            return k_attn_1_5_output == ""
        elif q_attn_1_5_output in {"4"}:
            return k_attn_1_5_output == "4"

    num_attn_2_1_pattern = select(attn_1_5_outputs, attn_1_5_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_5_output, attn_0_3_output):
        if attn_1_5_output in {"<s>", "2", "</s>", "0", "1", "4"}:
            return attn_0_3_output == ""
        elif attn_1_5_output in {"3"}:
            return attn_0_3_output == "3"

    num_attn_2_2_pattern = select(attn_0_3_outputs, attn_1_5_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_1_output, attn_0_0_output):
        if attn_1_1_output in {"<s>", "2", "3", "</s>", "0", "1", "4"}:
            return attn_0_0_output == ""

    num_attn_2_3_pattern = select(attn_0_0_outputs, attn_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_5_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_1_output, attn_0_3_output):
        if attn_1_1_output in {"<s>", "2", "3", "</s>", "0", "1", "4"}:
            return attn_0_3_output == ""

    num_attn_2_4_pattern = select(attn_0_3_outputs, attn_1_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_7_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_5_output, attn_0_3_output):
        if attn_1_5_output in {"<s>", "3", "</s>", "0", "4", "2"}:
            return attn_0_3_output == ""
        elif attn_1_5_output in {"1"}:
            return attn_0_3_output == "<s>"

    num_attn_2_5_pattern = select(attn_0_3_outputs, attn_1_5_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_5_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_0_output, attn_0_1_output):
        if attn_0_0_output in {"2", "3", "</s>", "0", "1", "4"}:
            return attn_0_1_output == ""
        elif attn_0_0_output in {"<s>"}:
            return attn_0_1_output == "<s>"

    num_attn_2_6_pattern = select(attn_0_1_outputs, attn_0_0_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_5_output, attn_1_7_output):
        if attn_1_5_output in {"0"}:
            return attn_1_7_output == "0"
        elif attn_1_5_output in {"<s>", "2", "3", "</s>", "1", "4"}:
            return attn_1_7_output == ""

    num_attn_2_7_pattern = select(attn_1_7_outputs, attn_1_5_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_0_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_4_output, attn_0_1_output):
        key = (attn_0_4_output, attn_0_1_output)
        if key in {
            ("0", "0"),
            ("1", "0"),
            ("2", "0"),
            ("3", "0"),
            ("4", "0"),
            ("<s>", "0"),
        }:
            return 34
        return 21

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_6_output, attn_0_4_output):
        key = (attn_0_6_output, attn_0_4_output)
        if key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "3"),
            ("0", "</s>"),
            ("2", "0"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "2"),
            ("4", "3"),
            ("4", "</s>"),
            ("</s>", "0"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "</s>"),
        }:
            return 9
        return 10

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_4_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 33

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_0_output, num_attn_1_6_output):
        key = (num_attn_1_0_output, num_attn_1_6_output)
        return 35

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
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
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
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
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
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
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
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


print(run(["<s>", "1", "0", "0", "2", "1", "2", "</s>"]))
