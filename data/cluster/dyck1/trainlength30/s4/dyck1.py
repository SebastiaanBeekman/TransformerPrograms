import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/dyck1/trainlength30/s4/dyck1_weights.csv",
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
        if q_position in {0, 45, 37}:
            return k_position == 40
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {8, 2, 29}:
            return k_position == 6
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {53, 5, 46}:
            return k_position == 39
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {56, 57, 58, 7}:
            return k_position == 46
        elif q_position in {9, 10}:
            return k_position == 8
        elif q_position in {11, 13}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {17, 41, 15}:
            return k_position == 14
        elif q_position in {16, 52, 21}:
            return k_position == 15
        elif q_position in {18, 19, 47}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {22}:
            return k_position == 20
        elif q_position in {30, 23}:
            return k_position == 22
        elif q_position in {24, 36}:
            return k_position == 23
        elif q_position in {25, 28, 38}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {31}:
            return k_position == 16
        elif q_position in {32}:
            return k_position == 38
        elif q_position in {48, 33}:
            return k_position == 53
        elif q_position in {34}:
            return k_position == 31
        elif q_position in {42, 35}:
            return k_position == 48
        elif q_position in {39}:
            return k_position == 36
        elif q_position in {40, 44}:
            return k_position == 21
        elif q_position in {43, 55}:
            return k_position == 51
        elif q_position in {49}:
            return k_position == 59
        elif q_position in {50, 59}:
            return k_position == 44
        elif q_position in {51}:
            return k_position == 58
        elif q_position in {54}:
            return k_position == 32

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 33}:
            return k_position == 38
        elif q_position in {1, 6}:
            return k_position == 5
        elif q_position in {2, 10}:
            return k_position == 8
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {38, 59, 5, 30}:
            return k_position == 11
        elif q_position in {15, 7}:
            return k_position == 14
        elif q_position in {8, 12}:
            return k_position == 6
        elif q_position in {48, 9}:
            return k_position == 30
        elif q_position in {11}:
            return k_position == 37
        elif q_position in {13}:
            return k_position == 59
        elif q_position in {43, 14, 39}:
            return k_position == 13
        elif q_position in {16, 18}:
            return k_position == 15
        elif q_position in {17, 19, 22}:
            return k_position == 16
        elif q_position in {32, 35, 37, 42, 44, 45, 46, 50, 20, 52}:
            return k_position == 17
        elif q_position in {21}:
            return k_position == 27
        elif q_position in {23}:
            return k_position == 34
        elif q_position in {24}:
            return k_position == 20
        elif q_position in {25, 27}:
            return k_position == 25
        elif q_position in {57, 26}:
            return k_position == 7
        elif q_position in {34, 28, 55, 31}:
            return k_position == 21
        elif q_position in {58, 29}:
            return k_position == 35
        elif q_position in {41, 51, 36}:
            return k_position == 19
        elif q_position in {40}:
            return k_position == 52
        elif q_position in {53, 47}:
            return k_position == 9
        elif q_position in {49}:
            return k_position == 50
        elif q_position in {54}:
            return k_position == 44
        elif q_position in {56}:
            return k_position == 54

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 48
        elif q_position in {1, 57}:
            return k_position == 39
        elif q_position in {2}:
            return k_position == 10
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 22
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {21, 7}:
            return k_position == 15
        elif q_position in {8, 10, 58, 22}:
            return k_position == 7
        elif q_position in {9, 50, 37, 46}:
            return k_position == 28
        elif q_position in {11}:
            return k_position == 6
        elif q_position in {12, 13}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {31, 15}:
            return k_position == 9
        elif q_position in {16, 30}:
            return k_position == 14
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18, 19, 42}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {28, 23}:
            return k_position == 21
        elif q_position in {24}:
            return k_position == 20
        elif q_position in {25, 27, 44, 29}:
            return k_position == 23
        elif q_position in {26, 43}:
            return k_position == 25
        elif q_position in {32}:
            return k_position == 18
        elif q_position in {33, 45}:
            return k_position == 54
        elif q_position in {34}:
            return k_position == 50
        elif q_position in {35}:
            return k_position == 52
        elif q_position in {51, 36}:
            return k_position == 47
        elif q_position in {38}:
            return k_position == 35
        elif q_position in {39}:
            return k_position == 36
        elif q_position in {40, 41}:
            return k_position == 53
        elif q_position in {47}:
            return k_position == 1
        elif q_position in {48}:
            return k_position == 40
        elif q_position in {49}:
            return k_position == 49
        elif q_position in {52}:
            return k_position == 32
        elif q_position in {53}:
            return k_position == 24
        elif q_position in {54}:
            return k_position == 57
        elif q_position in {55}:
            return k_position == 34
        elif q_position in {56, 59}:
            return k_position == 30

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 2

    attn_0_3_pattern = select_closest(positions, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 49, 53}:
            return k_position == 43
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {8, 2, 11}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4, 46}:
            return k_position == 2
        elif q_position in {18, 5, 54}:
            return k_position == 16
        elif q_position in {29, 6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 41
        elif q_position in {9, 20}:
            return k_position == 19
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {16, 17, 57, 15}:
            return k_position == 14
        elif q_position in {19, 21}:
            return k_position == 17
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 18
        elif q_position in {40, 55, 24, 26, 27}:
            return k_position == 23
        elif q_position in {25, 50}:
            return k_position == 24
        elif q_position in {28}:
            return k_position == 20
        elif q_position in {48, 56, 30}:
            return k_position == 58
        elif q_position in {31}:
            return k_position == 31
        elif q_position in {32}:
            return k_position == 30
        elif q_position in {33}:
            return k_position == 27
        elif q_position in {34}:
            return k_position == 32
        elif q_position in {35}:
            return k_position == 29
        elif q_position in {36}:
            return k_position == 10
        elif q_position in {37}:
            return k_position == 44
        elif q_position in {38}:
            return k_position == 59
        elif q_position in {41, 39}:
            return k_position == 53
        elif q_position in {42}:
            return k_position == 33
        elif q_position in {43}:
            return k_position == 5
        elif q_position in {44}:
            return k_position == 45
        elif q_position in {51, 45}:
            return k_position == 54
        elif q_position in {47}:
            return k_position == 57
        elif q_position in {52}:
            return k_position == 50
        elif q_position in {58}:
            return k_position == 36
        elif q_position in {59}:
            return k_position == 40

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 42, 50, 24, 26, 59, 30}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3, 29}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 29
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {39, 7}:
            return k_position == 36
        elif q_position in {8, 57}:
            return k_position == 7
        elif q_position in {48, 9, 12}:
            return k_position == 11
        elif q_position in {10, 13}:
            return k_position == 8
        elif q_position in {51, 11}:
            return k_position == 50
        elif q_position in {14, 15}:
            return k_position == 13
        elif q_position in {16, 17}:
            return k_position == 15
        elif q_position in {18, 23}:
            return k_position == 17
        elif q_position in {40, 19, 20, 21}:
            return k_position == 16
        elif q_position in {22}:
            return k_position == 18
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {27}:
            return k_position == 5
        elif q_position in {43, 28, 37}:
            return k_position == 27
        elif q_position in {41, 47, 31}:
            return k_position == 35
        elif q_position in {32}:
            return k_position == 34
        elif q_position in {33}:
            return k_position == 46
        elif q_position in {34}:
            return k_position == 51
        elif q_position in {35, 46}:
            return k_position == 58
        elif q_position in {36}:
            return k_position == 49
        elif q_position in {54, 52, 53, 38}:
            return k_position == 43
        elif q_position in {44}:
            return k_position == 25
        elif q_position in {45}:
            return k_position == 32
        elif q_position in {49}:
            return k_position == 55
        elif q_position in {55}:
            return k_position == 41
        elif q_position in {56}:
            return k_position == 21
        elif q_position in {58}:
            return k_position == 42

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 16, 35, 57}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 38
        elif q_position in {2, 10}:
            return k_position == 8
        elif q_position in {3, 5}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 43
        elif q_position in {8, 34, 48}:
            return k_position == 7
        elif q_position in {9, 11}:
            return k_position == 6
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {40, 42, 13, 38}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 13
        elif q_position in {17, 45, 30}:
            return k_position == 16
        elif q_position in {18, 19}:
            return k_position == 15
        elif q_position in {20}:
            return k_position == 18
        elif q_position in {59, 21, 23}:
            return k_position == 19
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {24, 28}:
            return k_position == 22
        elif q_position in {25}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 24
        elif q_position in {43, 27, 44, 53}:
            return k_position == 26
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {31}:
            return k_position == 47
        elif q_position in {32, 41}:
            return k_position == 1
        elif q_position in {33, 54}:
            return k_position == 41
        elif q_position in {36}:
            return k_position == 46
        elif q_position in {58, 37}:
            return k_position == 50
        elif q_position in {55, 39}:
            return k_position == 37
        elif q_position in {46}:
            return k_position == 34
        elif q_position in {49, 47}:
            return k_position == 9
        elif q_position in {50}:
            return k_position == 32
        elif q_position in {51}:
            return k_position == 36
        elif q_position in {52}:
            return k_position == 51
        elif q_position in {56}:
            return k_position == 54

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(token, position):
        if token in {"("}:
            return position == 17
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 2

    attn_0_7_pattern = select_closest(positions, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {
            0,
            2,
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
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
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
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif position in {1}:
            return token == "<s>"
        elif position in {35, 3, 4, 5}:
            return token == ")"
        elif position in {25}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"("}:
            return position == 47
        elif token in {")"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 17

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 41
        elif token in {")"}:
            return position == 36
        elif token in {"<s>"}:
            return position == 29

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {
            0,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            14,
            15,
            16,
            17,
            18,
            20,
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
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif position in {1, 3, 5}:
            return token == ")"
        elif position in {2, 13, 19, 21, 27}:
            return token == "<s>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0, 44, 29, 23}:
            return k_position == 45
        elif q_position in {1, 9}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 55
        elif q_position in {3, 4}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {10, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {8, 16}:
            return k_position == 6
        elif q_position in {11}:
            return k_position == 3
        elif q_position in {12}:
            return k_position == 12
        elif q_position in {13}:
            return k_position == 34
        elif q_position in {14}:
            return k_position == 14
        elif q_position in {51, 15}:
            return k_position == 47
        elif q_position in {17, 57, 55}:
            return k_position == 27
        elif q_position in {18}:
            return k_position == 51
        elif q_position in {19}:
            return k_position == 26
        elif q_position in {20}:
            return k_position == 43
        elif q_position in {21}:
            return k_position == 25
        elif q_position in {33, 22}:
            return k_position == 36
        elif q_position in {24, 25, 42, 35}:
            return k_position == 29
        elif q_position in {26}:
            return k_position == 48
        elif q_position in {56, 27}:
            return k_position == 44
        elif q_position in {48, 28}:
            return k_position == 31
        elif q_position in {30}:
            return k_position == 59
        elif q_position in {41, 31}:
            return k_position == 58
        elif q_position in {32}:
            return k_position == 57
        elif q_position in {34, 38}:
            return k_position == 56
        elif q_position in {36}:
            return k_position == 53
        elif q_position in {59, 37}:
            return k_position == 21
        elif q_position in {39}:
            return k_position == 39
        elif q_position in {40, 46}:
            return k_position == 17
        elif q_position in {43, 53}:
            return k_position == 54
        elif q_position in {45, 54}:
            return k_position == 38
        elif q_position in {47}:
            return k_position == 40
        elif q_position in {49}:
            return k_position == 32
        elif q_position in {50, 52}:
            return k_position == 19
        elif q_position in {58}:
            return k_position == 22

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"("}:
            return position == 8
        elif token in {")", "<s>"}:
            return position == 55

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {
            0,
            5,
            6,
            7,
            8,
            9,
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
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif position in {1, 2, 3, 4, 10}:
            return token == ")"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {
            0,
            6,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            18,
            19,
            20,
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
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif position in {1, 29}:
            return token == "<s>"
        elif position in {2, 3, 4, 5, 7, 9}:
            return token == ")"
        elif position in {17, 21}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_7_output):
        key = (attn_0_0_output, attn_0_7_output)
        if key in {(")", "("), (")", ")"), (")", "<s>")}:
            return 58
        return 1

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_7_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_2_output):
        key = (attn_0_6_output, attn_0_2_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 2
        elif key in {(")", "("), ("<s>", "(")}:
            return 52
        return 43

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_2_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_5_output):
        key = (num_attn_0_0_output, num_attn_0_5_output)
        return 18

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_4_output):
        key = (num_attn_0_1_output, num_attn_0_4_output)
        return 59

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 1, 27}:
            return position == 5
        elif mlp_0_0_output in {32, 2, 4, 6, 8, 13, 45, 15, 16, 20}:
            return position == 27
        elif mlp_0_0_output in {41, 3, 52, 43}:
            return position == 1
        elif mlp_0_0_output in {5, 30}:
            return position == 25
        elif mlp_0_0_output in {10, 36, 7}:
            return position == 58
        elif mlp_0_0_output in {9}:
            return position == 56
        elif mlp_0_0_output in {11, 37}:
            return position == 55
        elif mlp_0_0_output in {12}:
            return position == 34
        elif mlp_0_0_output in {14}:
            return position == 54
        elif mlp_0_0_output in {17, 21, 57}:
            return position == 9
        elif mlp_0_0_output in {18}:
            return position == 20
        elif mlp_0_0_output in {35, 19}:
            return position == 50
        elif mlp_0_0_output in {22}:
            return position == 10
        elif mlp_0_0_output in {23}:
            return position == 44
        elif mlp_0_0_output in {24}:
            return position == 19
        elif mlp_0_0_output in {25, 34}:
            return position == 18
        elif mlp_0_0_output in {26, 54}:
            return position == 17
        elif mlp_0_0_output in {58, 28}:
            return position == 6
        elif mlp_0_0_output in {29, 47}:
            return position == 36
        elif mlp_0_0_output in {31}:
            return position == 11
        elif mlp_0_0_output in {33, 59}:
            return position == 21
        elif mlp_0_0_output in {48, 56, 38, 55}:
            return position == 23
        elif mlp_0_0_output in {39}:
            return position == 37
        elif mlp_0_0_output in {40}:
            return position == 39
        elif mlp_0_0_output in {42}:
            return position == 15
        elif mlp_0_0_output in {44}:
            return position == 59
        elif mlp_0_0_output in {46}:
            return position == 49
        elif mlp_0_0_output in {49}:
            return position == 12
        elif mlp_0_0_output in {50}:
            return position == 31
        elif mlp_0_0_output in {51}:
            return position == 46
        elif mlp_0_0_output in {53}:
            return position == 3

    attn_1_0_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0}:
            return mlp_0_1_output == 30
        elif mlp_0_0_output in {1, 36, 39, 41, 43, 13, 58, 31}:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {2, 22}:
            return mlp_0_1_output == 20
        elif mlp_0_0_output in {3, 44}:
            return mlp_0_1_output == 13
        elif mlp_0_0_output in {4}:
            return mlp_0_1_output == 26
        elif mlp_0_0_output in {34, 5, 10, 14, 15, 17, 19, 25, 26}:
            return mlp_0_1_output == 31
        elif mlp_0_0_output in {21, 6}:
            return mlp_0_1_output == 45
        elif mlp_0_0_output in {7}:
            return mlp_0_1_output == 49
        elif mlp_0_0_output in {8, 59}:
            return mlp_0_1_output == 39
        elif mlp_0_0_output in {9}:
            return mlp_0_1_output == 46
        elif mlp_0_0_output in {11}:
            return mlp_0_1_output == 21
        elif mlp_0_0_output in {27, 12, 37}:
            return mlp_0_1_output == 33
        elif mlp_0_0_output in {16, 23}:
            return mlp_0_1_output == 40
        elif mlp_0_0_output in {18, 29}:
            return mlp_0_1_output == 12
        elif mlp_0_0_output in {20}:
            return mlp_0_1_output == 57
        elif mlp_0_0_output in {24, 53, 45}:
            return mlp_0_1_output == 28
        elif mlp_0_0_output in {50, 35, 28}:
            return mlp_0_1_output == 48
        elif mlp_0_0_output in {30}:
            return mlp_0_1_output == 16
        elif mlp_0_0_output in {32, 47}:
            return mlp_0_1_output == 15
        elif mlp_0_0_output in {33}:
            return mlp_0_1_output == 5
        elif mlp_0_0_output in {51, 38}:
            return mlp_0_1_output == 35
        elif mlp_0_0_output in {40}:
            return mlp_0_1_output == 55
        elif mlp_0_0_output in {42}:
            return mlp_0_1_output == 29
        elif mlp_0_0_output in {46}:
            return mlp_0_1_output == 19
        elif mlp_0_0_output in {48}:
            return mlp_0_1_output == 34
        elif mlp_0_0_output in {49}:
            return mlp_0_1_output == 24
        elif mlp_0_0_output in {52}:
            return mlp_0_1_output == 43
        elif mlp_0_0_output in {54}:
            return mlp_0_1_output == 51
        elif mlp_0_0_output in {55}:
            return mlp_0_1_output == 44
        elif mlp_0_0_output in {56}:
            return mlp_0_1_output == 37
        elif mlp_0_0_output in {57}:
            return mlp_0_1_output == 27

    attn_1_1_pattern = select_closest(mlp_0_1_outputs, mlp_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_4_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_6_output, position):
        if attn_0_6_output in {"("}:
            return position == 1
        elif attn_0_6_output in {")"}:
            return position == 38
        elif attn_0_6_output in {"<s>"}:
            return position == 5

    attn_1_2_pattern = select_closest(positions, attn_0_6_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 9

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_4_output, mlp_0_1_output):
        if attn_0_4_output in {"("}:
            return mlp_0_1_output == 48
        elif attn_0_4_output in {")"}:
            return mlp_0_1_output == 2
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_1_output == 43

    attn_1_4_pattern = select_closest(mlp_0_1_outputs, attn_0_4_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 40, 43}:
            return position == 44
        elif mlp_0_0_output in {1, 3, 44, 41}:
            return position == 1
        elif mlp_0_0_output in {2, 18, 50, 7}:
            return position == 45
        elif mlp_0_0_output in {11, 4}:
            return position == 37
        elif mlp_0_0_output in {19, 10, 13, 5}:
            return position == 29
        elif mlp_0_0_output in {6, 8, 12, 49, 52}:
            return position == 27
        elif mlp_0_0_output in {9, 37}:
            return position == 34
        elif mlp_0_0_output in {14}:
            return position == 47
        elif mlp_0_0_output in {17, 15}:
            return position == 39
        elif mlp_0_0_output in {34, 56, 16, 21, 23, 24, 25, 26, 27, 29, 57}:
            return position == 5
        elif mlp_0_0_output in {20}:
            return position == 53
        elif mlp_0_0_output in {22}:
            return position == 3
        elif mlp_0_0_output in {28}:
            return position == 11
        elif mlp_0_0_output in {30, 55}:
            return position == 7
        elif mlp_0_0_output in {31}:
            return position == 6
        elif mlp_0_0_output in {32, 48}:
            return position == 21
        elif mlp_0_0_output in {33, 59, 38, 47}:
            return position == 25
        elif mlp_0_0_output in {35}:
            return position == 57
        elif mlp_0_0_output in {36, 45}:
            return position == 56
        elif mlp_0_0_output in {39}:
            return position == 32
        elif mlp_0_0_output in {42}:
            return position == 23
        elif mlp_0_0_output in {46}:
            return position == 52
        elif mlp_0_0_output in {51}:
            return position == 58
        elif mlp_0_0_output in {53}:
            return position == 49
        elif mlp_0_0_output in {54}:
            return position == 24
        elif mlp_0_0_output in {58}:
            return position == 9

    attn_1_5_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_3_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(mlp_0_0_output, position):
        if mlp_0_0_output in {0}:
            return position == 17
        elif mlp_0_0_output in {1, 50, 3, 41}:
            return position == 1
        elif mlp_0_0_output in {2, 6, 40, 9, 10, 20}:
            return position == 27
        elif mlp_0_0_output in {59, 4, 5, 55}:
            return position == 23
        elif mlp_0_0_output in {33, 36, 7, 16, 48, 52, 57, 28, 30}:
            return position == 25
        elif mlp_0_0_output in {8, 32, 38}:
            return position == 15
        elif mlp_0_0_output in {35, 11}:
            return position == 33
        elif mlp_0_0_output in {25, 12}:
            return position == 42
        elif mlp_0_0_output in {13}:
            return position == 29
        elif mlp_0_0_output in {18, 14}:
            return position == 37
        elif mlp_0_0_output in {15}:
            return position == 40
        elif mlp_0_0_output in {17}:
            return position == 47
        elif mlp_0_0_output in {19}:
            return position == 50
        elif mlp_0_0_output in {21}:
            return position == 13
        elif mlp_0_0_output in {26, 22}:
            return position == 19
        elif mlp_0_0_output in {23}:
            return position == 39
        elif mlp_0_0_output in {24, 58, 44}:
            return position == 7
        elif mlp_0_0_output in {34, 54, 56, 27, 29}:
            return position == 5
        elif mlp_0_0_output in {31}:
            return position == 3
        elif mlp_0_0_output in {37}:
            return position == 12
        elif mlp_0_0_output in {45, 39}:
            return position == 46
        elif mlp_0_0_output in {42}:
            return position == 14
        elif mlp_0_0_output in {43}:
            return position == 51
        elif mlp_0_0_output in {46}:
            return position == 16
        elif mlp_0_0_output in {47}:
            return position == 45
        elif mlp_0_0_output in {49}:
            return position == 26
        elif mlp_0_0_output in {51}:
            return position == 35
        elif mlp_0_0_output in {53}:
            return position == 11

    attn_1_6_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 53}:
            return position == 52
        elif mlp_0_0_output in {1, 3, 52, 41}:
            return position == 1
        elif mlp_0_0_output in {2, 36, 13, 15, 16, 20, 23, 26, 28}:
            return position == 27
        elif mlp_0_0_output in {4, 38, 54, 55, 59}:
            return position == 23
        elif mlp_0_0_output in {32, 35, 5, 6, 10, 11, 12, 47, 49, 19, 57, 30}:
            return position == 25
        elif mlp_0_0_output in {7}:
            return position == 46
        elif mlp_0_0_output in {8}:
            return position == 28
        elif mlp_0_0_output in {9, 22}:
            return position == 31
        elif mlp_0_0_output in {14}:
            return position == 47
        elif mlp_0_0_output in {17}:
            return position == 33
        elif mlp_0_0_output in {25, 18, 27}:
            return position == 3
        elif mlp_0_0_output in {43, 21}:
            return position == 38
        elif mlp_0_0_output in {24}:
            return position == 51
        elif mlp_0_0_output in {29}:
            return position == 48
        elif mlp_0_0_output in {31}:
            return position == 17
        elif mlp_0_0_output in {33}:
            return position == 19
        elif mlp_0_0_output in {34}:
            return position == 37
        elif mlp_0_0_output in {37}:
            return position == 44
        elif mlp_0_0_output in {58, 39}:
            return position == 5
        elif mlp_0_0_output in {40}:
            return position == 49
        elif mlp_0_0_output in {48, 42}:
            return position == 21
        elif mlp_0_0_output in {44}:
            return position == 53
        elif mlp_0_0_output in {45}:
            return position == 11
        elif mlp_0_0_output in {46}:
            return position == 0
        elif mlp_0_0_output in {50}:
            return position == 55
        elif mlp_0_0_output in {51}:
            return position == 39
        elif mlp_0_0_output in {56}:
            return position == 9

    attn_1_7_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 5}:
            return mlp_0_1_output == 24
        elif mlp_0_0_output in {1, 33, 4, 11, 44}:
            return mlp_0_1_output == 14
        elif mlp_0_0_output in {2}:
            return mlp_0_1_output == 39
        elif mlp_0_0_output in {3}:
            return mlp_0_1_output == 6
        elif mlp_0_0_output in {6}:
            return mlp_0_1_output == 10
        elif mlp_0_0_output in {7}:
            return mlp_0_1_output == 9
        elif mlp_0_0_output in {8}:
            return mlp_0_1_output == 7
        elif mlp_0_0_output in {9, 31}:
            return mlp_0_1_output == 52
        elif mlp_0_0_output in {41, 10, 36}:
            return mlp_0_1_output == 55
        elif mlp_0_0_output in {12}:
            return mlp_0_1_output == 11
        elif mlp_0_0_output in {13}:
            return mlp_0_1_output == 29
        elif mlp_0_0_output in {14}:
            return mlp_0_1_output == 42
        elif mlp_0_0_output in {22, 15}:
            return mlp_0_1_output == 30
        elif mlp_0_0_output in {16}:
            return mlp_0_1_output == 50
        elif mlp_0_0_output in {40, 48, 17, 18, 50, 21, 26}:
            return mlp_0_1_output == 36
        elif mlp_0_0_output in {43, 19, 20, 57}:
            return mlp_0_1_output == 19
        elif mlp_0_0_output in {25, 23}:
            return mlp_0_1_output == 13
        elif mlp_0_0_output in {24}:
            return mlp_0_1_output == 25
        elif mlp_0_0_output in {27}:
            return mlp_0_1_output == 8
        elif mlp_0_0_output in {28}:
            return mlp_0_1_output == 43
        elif mlp_0_0_output in {53, 29, 38}:
            return mlp_0_1_output == 45
        elif mlp_0_0_output in {30}:
            return mlp_0_1_output == 48
        elif mlp_0_0_output in {32, 42, 45}:
            return mlp_0_1_output == 16
        elif mlp_0_0_output in {34}:
            return mlp_0_1_output == 23
        elif mlp_0_0_output in {35, 54}:
            return mlp_0_1_output == 12
        elif mlp_0_0_output in {37}:
            return mlp_0_1_output == 26
        elif mlp_0_0_output in {39}:
            return mlp_0_1_output == 46
        elif mlp_0_0_output in {58, 46}:
            return mlp_0_1_output == 1
        elif mlp_0_0_output in {47}:
            return mlp_0_1_output == 35
        elif mlp_0_0_output in {49}:
            return mlp_0_1_output == 54
        elif mlp_0_0_output in {51}:
            return mlp_0_1_output == 58
        elif mlp_0_0_output in {52}:
            return mlp_0_1_output == 4
        elif mlp_0_0_output in {55}:
            return mlp_0_1_output == 59
        elif mlp_0_0_output in {56}:
            return mlp_0_1_output == 57
        elif mlp_0_0_output in {59}:
            return mlp_0_1_output == 15

    num_attn_1_0_pattern = select(mlp_0_1_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_5_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, mlp_0_1_output):
        if position in {0, 37, 6, 7, 8, 38, 48, 20, 59}:
            return mlp_0_1_output == 43
        elif position in {1, 52, 55}:
            return mlp_0_1_output == 36
        elif position in {
            2,
            3,
            9,
            11,
            12,
            13,
            14,
            15,
            18,
            24,
            26,
            28,
            33,
            34,
            35,
            45,
            49,
            51,
            54,
            57,
        }:
            return mlp_0_1_output == 52
        elif position in {4}:
            return mlp_0_1_output == 11
        elif position in {5}:
            return mlp_0_1_output == 16
        elif position in {10}:
            return mlp_0_1_output == 34
        elif position in {16}:
            return mlp_0_1_output == 23
        elif position in {17, 43}:
            return mlp_0_1_output == 10
        elif position in {19, 36, 21, 46}:
            return mlp_0_1_output == 30
        elif position in {22}:
            return mlp_0_1_output == 18
        elif position in {23}:
            return mlp_0_1_output == 8
        elif position in {25, 39}:
            return mlp_0_1_output == 6
        elif position in {27}:
            return mlp_0_1_output == 13
        elif position in {29}:
            return mlp_0_1_output == 31
        elif position in {32, 30}:
            return mlp_0_1_output == 1
        elif position in {42, 47, 31}:
            return mlp_0_1_output == 41
        elif position in {40}:
            return mlp_0_1_output == 55
        elif position in {41}:
            return mlp_0_1_output == 40
        elif position in {44}:
            return mlp_0_1_output == 9
        elif position in {50}:
            return mlp_0_1_output == 45
        elif position in {53}:
            return mlp_0_1_output == 24
        elif position in {56}:
            return mlp_0_1_output == 15
        elif position in {58}:
            return mlp_0_1_output == 32

    num_attn_1_1_pattern = select(mlp_0_1_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_5_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 10, 46, 21, 57}:
            return position == 10
        elif num_mlp_0_0_output in {1, 12, 49, 33}:
            return position == 13
        elif num_mlp_0_0_output in {56, 2, 20}:
            return position == 20
        elif num_mlp_0_0_output in {3, 35, 37, 8, 59, 11, 23, 27}:
            return position == 11
        elif num_mlp_0_0_output in {42, 4}:
            return position == 19
        elif num_mlp_0_0_output in {5, 14}:
            return position == 1
        elif num_mlp_0_0_output in {16, 58, 6, 55}:
            return position == 15
        elif num_mlp_0_0_output in {7}:
            return position == 18
        elif num_mlp_0_0_output in {9, 43, 44, 45, 15, 48, 18, 50, 22, 54, 26}:
            return position == 12
        elif num_mlp_0_0_output in {24, 17, 13}:
            return position == 14
        elif num_mlp_0_0_output in {34, 19, 30, 39}:
            return position == 9
        elif num_mlp_0_0_output in {25, 38}:
            return position == 17
        elif num_mlp_0_0_output in {28}:
            return position == 41
        elif num_mlp_0_0_output in {29}:
            return position == 31
        elif num_mlp_0_0_output in {31}:
            return position == 22
        elif num_mlp_0_0_output in {32}:
            return position == 28
        elif num_mlp_0_0_output in {36}:
            return position == 25
        elif num_mlp_0_0_output in {40}:
            return position == 59
        elif num_mlp_0_0_output in {41}:
            return position == 23
        elif num_mlp_0_0_output in {47}:
            return position == 16
        elif num_mlp_0_0_output in {51}:
            return position == 5
        elif num_mlp_0_0_output in {52}:
            return position == 55
        elif num_mlp_0_0_output in {53}:
            return position == 21

    num_attn_1_2_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_5_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_5_output, mlp_0_0_output):
        if attn_0_5_output in {"(", "<s>"}:
            return mlp_0_0_output == 31
        elif attn_0_5_output in {")"}:
            return mlp_0_0_output == 58

    num_attn_1_3_pattern = select(mlp_0_0_outputs, attn_0_5_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(num_mlp_0_0_output, attn_0_6_output):
        if num_mlp_0_0_output in {0, 35, 37, 40, 11, 45, 46, 51}:
            return attn_0_6_output == "("
        elif num_mlp_0_0_output in {
            1,
            2,
            4,
            5,
            6,
            9,
            10,
            12,
            14,
            17,
            20,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            30,
            32,
            33,
            34,
            38,
            42,
            44,
            47,
            48,
            49,
            50,
            54,
            55,
            56,
            58,
            59,
        }:
            return attn_0_6_output == ""
        elif num_mlp_0_0_output in {
            3,
            36,
            7,
            8,
            39,
            41,
            43,
            13,
            15,
            16,
            18,
            19,
            52,
            53,
            25,
            57,
            29,
            31,
        }:
            return attn_0_6_output == ")"

    num_attn_1_4_pattern = select(
        attn_0_6_outputs, num_mlp_0_0_outputs, num_predicate_1_4
    )
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_0_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_4_output, attn_0_5_output):
        if attn_0_4_output in {"(", ")"}:
            return attn_0_5_output == ""
        elif attn_0_4_output in {"<s>"}:
            return attn_0_5_output == ")"

    num_attn_1_5_pattern = select(attn_0_5_outputs, attn_0_4_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"(", ")", "<s>"}:
            return mlp_0_0_output == 31

    num_attn_1_6_pattern = select(mlp_0_0_outputs, attn_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_7_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(num_mlp_0_0_output, mlp_0_1_output):
        if num_mlp_0_0_output in {
            0,
            1,
            3,
            5,
            8,
            9,
            10,
            12,
            14,
            15,
            16,
            18,
            19,
            20,
            22,
            23,
            25,
            27,
            29,
            30,
            33,
            35,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            45,
            46,
            47,
            48,
            49,
            50,
            52,
            55,
            56,
            59,
        }:
            return mlp_0_1_output == 43
        elif num_mlp_0_0_output in {2}:
            return mlp_0_1_output == 18
        elif num_mlp_0_0_output in {4, 6}:
            return mlp_0_1_output == 31
        elif num_mlp_0_0_output in {44, 7}:
            return mlp_0_1_output == 36
        elif num_mlp_0_0_output in {11}:
            return mlp_0_1_output == 7
        elif num_mlp_0_0_output in {13}:
            return mlp_0_1_output == 48
        elif num_mlp_0_0_output in {17, 53, 57, 58, 31}:
            return mlp_0_1_output == 41
        elif num_mlp_0_0_output in {21}:
            return mlp_0_1_output == 33
        elif num_mlp_0_0_output in {24}:
            return mlp_0_1_output == 27
        elif num_mlp_0_0_output in {26}:
            return mlp_0_1_output == 39
        elif num_mlp_0_0_output in {28}:
            return mlp_0_1_output == 37
        elif num_mlp_0_0_output in {32, 51}:
            return mlp_0_1_output == 40
        elif num_mlp_0_0_output in {34}:
            return mlp_0_1_output == 47
        elif num_mlp_0_0_output in {36}:
            return mlp_0_1_output == 11
        elif num_mlp_0_0_output in {54}:
            return mlp_0_1_output == 0

    num_attn_1_7_pattern = select(
        mlp_0_1_outputs, num_mlp_0_0_outputs, num_predicate_1_7
    )
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_5_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, attn_1_7_output):
        key = (attn_1_3_output, attn_1_7_output)
        if key in {(")", ")")}:
            return 26
        elif key in {("<s>", "(")}:
            return 57
        elif key in {("<s>", ")")}:
            return 2
        return 41

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_7_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {("<s>", "<s>")}:
            return 2
        return 53

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output, num_attn_1_2_output):
        key = (num_attn_1_4_output, num_attn_1_2_output)
        return 54

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_5_output, num_attn_1_0_output):
        key = (num_attn_1_5_output, num_attn_1_0_output)
        return 17

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"("}:
            return position == 9
        elif token in {")"}:
            return position == 13
        elif token in {"<s>"}:
            return position == 5

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_7_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_3_output, position):
        if attn_0_3_output in {"("}:
            return position == 3
        elif attn_0_3_output in {")"}:
            return position == 11
        elif attn_0_3_output in {"<s>"}:
            return position == 6

    attn_2_1_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_7_output, position):
        if attn_0_7_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_7_output in {")"}:
            return position == 15

    attn_2_2_pattern = select_closest(positions, attn_0_7_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_0_output, mlp_0_0_output):
        if attn_0_0_output in {"("}:
            return mlp_0_0_output == 3
        elif attn_0_0_output in {")"}:
            return mlp_0_0_output == 31
        elif attn_0_0_output in {"<s>"}:
            return mlp_0_0_output == 5

    attn_2_3_pattern = select_closest(mlp_0_0_outputs, attn_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_6_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 3
        elif attn_0_0_output in {")"}:
            return position == 11
        elif attn_0_0_output in {"<s>"}:
            return position == 1

    attn_2_4_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(token, mlp_0_0_output):
        if token in {"(", ")"}:
            return mlp_0_0_output == 31
        elif token in {"<s>"}:
            return mlp_0_0_output == 5

    attn_2_5_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_1_5_output, position):
        if attn_1_5_output in {"(", "<s>"}:
            return position == 3
        elif attn_1_5_output in {")"}:
            return position == 4

    attn_2_6_pattern = select_closest(positions, attn_1_5_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_7_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(token, mlp_0_1_output):
        if token in {"(", ")"}:
            return mlp_0_1_output == 2
        elif token in {"<s>"}:
            return mlp_0_1_output == 56

    attn_2_7_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_0_output, attn_0_3_output):
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
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return attn_0_3_output == ""

    num_attn_2_0_pattern = select(attn_0_3_outputs, mlp_0_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_5_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_2_output, mlp_0_0_output):
        if attn_1_2_output in {"(", "<s>"}:
            return mlp_0_0_output == 31
        elif attn_1_2_output in {")"}:
            return mlp_0_0_output == 58

    num_attn_2_1_pattern = select(mlp_0_0_outputs, attn_1_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_4_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_attn_1_7_output, k_attn_1_7_output):
        if q_attn_1_7_output in {"(", ")", "<s>"}:
            return k_attn_1_7_output == ""

    num_attn_2_2_pattern = select(attn_1_7_outputs, attn_1_7_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_5_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_1_0_output, mlp_0_1_output):
        if mlp_1_0_output in {0, 20}:
            return mlp_0_1_output == 8
        elif mlp_1_0_output in {1, 3, 38, 7, 8, 52, 21, 56}:
            return mlp_0_1_output == 31
        elif mlp_1_0_output in {
            2,
            4,
            5,
            6,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            24,
            26,
            29,
            32,
            33,
            34,
            36,
            37,
            41,
            42,
            43,
            46,
            47,
            48,
            49,
            50,
            51,
            55,
            57,
            58,
            59,
        }:
            return mlp_0_1_output == 2
        elif mlp_1_0_output in {10, 35, 22}:
            return mlp_0_1_output == 20
        elif mlp_1_0_output in {23}:
            return mlp_0_1_output == 27
        elif mlp_1_0_output in {25}:
            return mlp_0_1_output == 53
        elif mlp_1_0_output in {27}:
            return mlp_0_1_output == 56
        elif mlp_1_0_output in {28}:
            return mlp_0_1_output == 17
        elif mlp_1_0_output in {30}:
            return mlp_0_1_output == 13
        elif mlp_1_0_output in {31}:
            return mlp_0_1_output == 58
        elif mlp_1_0_output in {39}:
            return mlp_0_1_output == 57
        elif mlp_1_0_output in {40}:
            return mlp_0_1_output == 40
        elif mlp_1_0_output in {44}:
            return mlp_0_1_output == 4
        elif mlp_1_0_output in {45, 54}:
            return mlp_0_1_output == 50
        elif mlp_1_0_output in {53}:
            return mlp_0_1_output == 34

    num_attn_2_3_pattern = select(mlp_0_1_outputs, mlp_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_4_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_4_output, mlp_1_0_output):
        if attn_1_4_output in {"(", "<s>"}:
            return mlp_1_0_output == 2
        elif attn_1_4_output in {")"}:
            return mlp_1_0_output == 12

    num_attn_2_4_pattern = select(mlp_1_0_outputs, attn_1_4_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_3_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_5_output, attn_1_4_output):
        if attn_1_5_output in {"(", ")", "<s>"}:
            return attn_1_4_output == ""

    num_attn_2_5_pattern = select(attn_1_4_outputs, attn_1_5_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_1_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(num_mlp_0_0_output, attn_1_3_output):
        if num_mlp_0_0_output in {
            0,
            3,
            5,
            6,
            8,
            9,
            15,
            16,
            18,
            19,
            21,
            22,
            24,
            25,
            27,
            28,
            30,
            31,
            32,
            34,
            35,
            36,
            38,
            40,
            42,
            43,
            44,
            46,
            47,
            49,
            51,
            53,
            54,
            55,
            57,
            59,
        }:
            return attn_1_3_output == ")"
        elif num_mlp_0_0_output in {
            1,
            2,
            33,
            58,
            7,
            39,
            10,
            11,
            12,
            45,
            48,
            20,
            56,
            26,
            29,
        }:
            return attn_1_3_output == ""
        elif num_mlp_0_0_output in {4, 37, 41, 13, 14, 17, 50, 52, 23}:
            return attn_1_3_output == "<s>"

    num_attn_2_6_pattern = select(
        attn_1_3_outputs, num_mlp_0_0_outputs, num_predicate_2_6
    )
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_7_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_1_output, mlp_0_0_output):
        if attn_1_1_output in {"(", "<s>"}:
            return mlp_0_0_output == 31
        elif attn_1_1_output in {")"}:
            return mlp_0_0_output == 53

    num_attn_2_7_pattern = select(mlp_0_0_outputs, attn_1_1_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_0_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_2_5_output):
        key = (attn_2_1_output, attn_2_5_output)
        if key in {("(", "("), ("(", ")"), ("(", "<s>")}:
            return 58
        return 52

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_5_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, attn_2_6_output):
        key = (attn_2_2_output, attn_2_6_output)
        if key in {(")", "<s>"), ("<s>", "<s>")}:
            return 10
        elif key in {(")", ")"), ("<s>", ")")}:
            return 36
        return 22

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_6_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_2_output, num_attn_1_6_output):
        key = (num_attn_2_2_output, num_attn_1_6_output)
        return 15

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_2_output, num_attn_1_0_output):
        key = (num_attn_2_2_output, num_attn_1_0_output)
        return 51

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_0_outputs)
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


print(
    run(
        [
            "<s>",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            ")",
            ")",
        ]
    )
)
