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
        "output/length/rasp/dyck1/trainlength30/s3/dyck1_weights.csv",
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
        if q_position in {0, 37, 31}:
            return k_position == 56
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 12, 20}:
            return k_position == 10
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4, 5}:
            return k_position == 2
        elif q_position in {52, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 20
        elif q_position in {8, 29}:
            return k_position == 4
        elif q_position in {9}:
            return k_position == 42
        elif q_position in {24, 10, 26, 53}:
            return k_position == 7
        elif q_position in {11, 54}:
            return k_position == 46
        elif q_position in {22, 13, 14, 15}:
            return k_position == 9
        elif q_position in {16, 32}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 11
        elif q_position in {18}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 6
        elif q_position in {21}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 52
        elif q_position in {40, 25, 27, 44}:
            return k_position == 32
        elif q_position in {28}:
            return k_position == 19
        elif q_position in {58, 30}:
            return k_position == 58
        elif q_position in {33, 51}:
            return k_position == 30
        elif q_position in {34}:
            return k_position == 21
        elif q_position in {35}:
            return k_position == 54
        elif q_position in {36}:
            return k_position == 34
        elif q_position in {59, 38, 55}:
            return k_position == 48
        elif q_position in {39}:
            return k_position == 25
        elif q_position in {41}:
            return k_position == 38
        elif q_position in {48, 42}:
            return k_position == 31
        elif q_position in {43}:
            return k_position == 51
        elif q_position in {45}:
            return k_position == 36
        elif q_position in {46}:
            return k_position == 35
        elif q_position in {47}:
            return k_position == 37
        elif q_position in {49}:
            return k_position == 49
        elif q_position in {50}:
            return k_position == 41
        elif q_position in {56}:
            return k_position == 55
        elif q_position in {57}:
            return k_position == 43

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 57}:
            return k_position == 54
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 10}:
            return k_position == 8
        elif q_position in {29, 3, 5}:
            return k_position == 3
        elif q_position in {27, 4}:
            return k_position == 2
        elif q_position in {8, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {9}:
            return k_position == 38
        elif q_position in {11}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 13
        elif q_position in {16, 50, 52}:
            return k_position == 14
        elif q_position in {17, 19}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 16
        elif q_position in {20}:
            return k_position == 18
        elif q_position in {21, 23}:
            return k_position == 19
        elif q_position in {22}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 7
        elif q_position in {25}:
            return k_position == 21
        elif q_position in {26}:
            return k_position == 24
        elif q_position in {28}:
            return k_position == 26
        elif q_position in {54, 30}:
            return k_position == 40
        elif q_position in {31}:
            return k_position == 32
        elif q_position in {32}:
            return k_position == 27
        elif q_position in {33}:
            return k_position == 39
        elif q_position in {34, 37, 39}:
            return k_position == 56
        elif q_position in {35}:
            return k_position == 52
        elif q_position in {36}:
            return k_position == 47
        elif q_position in {38}:
            return k_position == 49
        elif q_position in {40}:
            return k_position == 22
        elif q_position in {41, 51, 49}:
            return k_position == 29
        elif q_position in {56, 42}:
            return k_position == 23
        elif q_position in {43}:
            return k_position == 41
        elif q_position in {44}:
            return k_position == 48
        elif q_position in {45}:
            return k_position == 50
        elif q_position in {46}:
            return k_position == 59
        elif q_position in {47}:
            return k_position == 55
        elif q_position in {48}:
            return k_position == 31
        elif q_position in {53}:
            return k_position == 36
        elif q_position in {55}:
            return k_position == 37
        elif q_position in {58}:
            return k_position == 43
        elif q_position in {59}:
            return k_position == 46

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(token, position):
        if token in {"("}:
            return position == 7
        elif token in {")"}:
            return position == 22
        elif token in {"<s>"}:
            return position == 2

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 1
        elif q_position in {2, 38}:
            return k_position == 54
        elif q_position in {57, 42, 3, 50}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 4
        elif q_position in {9, 7}:
            return k_position == 6
        elif q_position in {8, 53}:
            return k_position == 7
        elif q_position in {10, 39}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13, 47}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {32, 19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {44, 21, 52}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {59, 43, 30}:
            return k_position == 47
        elif q_position in {31}:
            return k_position == 58
        elif q_position in {33, 37}:
            return k_position == 57
        elif q_position in {34}:
            return k_position == 51
        elif q_position in {58, 35}:
            return k_position == 46
        elif q_position in {56, 41, 36}:
            return k_position == 42
        elif q_position in {40}:
            return k_position == 5
        elif q_position in {45}:
            return k_position == 39
        elif q_position in {46}:
            return k_position == 36
        elif q_position in {48}:
            return k_position == 52
        elif q_position in {49}:
            return k_position == 56
        elif q_position in {51}:
            return k_position == 59
        elif q_position in {54, 55}:
            return k_position == 55

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 21

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 37
        elif q_position in {1, 59}:
            return k_position == 47
        elif q_position in {2}:
            return k_position == 20
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {27, 4, 23}:
            return k_position == 2
        elif q_position in {35, 5}:
            return k_position == 45
        elif q_position in {6, 7}:
            return k_position == 5
        elif q_position in {8, 9, 20, 53, 25}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {11, 28, 14}:
            return k_position == 9
        elif q_position in {12, 15}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {16}:
            return k_position == 13
        elif q_position in {24, 17, 26}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 4
        elif q_position in {21}:
            return k_position == 17
        elif q_position in {22}:
            return k_position == 6
        elif q_position in {29}:
            return k_position == 14
        elif q_position in {56, 50, 30}:
            return k_position == 38
        elif q_position in {48, 51, 31}:
            return k_position == 56
        elif q_position in {32, 46}:
            return k_position == 29
        elif q_position in {33}:
            return k_position == 43
        elif q_position in {34, 39}:
            return k_position == 34
        elif q_position in {42, 36}:
            return k_position == 31
        elif q_position in {37}:
            return k_position == 59
        elif q_position in {38}:
            return k_position == 44
        elif q_position in {40}:
            return k_position == 1
        elif q_position in {41}:
            return k_position == 42
        elif q_position in {43}:
            return k_position == 36
        elif q_position in {44, 55}:
            return k_position == 50
        elif q_position in {45}:
            return k_position == 58
        elif q_position in {47}:
            return k_position == 25
        elif q_position in {49}:
            return k_position == 53
        elif q_position in {57, 52}:
            return k_position == 41
        elif q_position in {54}:
            return k_position == 21
        elif q_position in {58}:
            return k_position == 49

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 32, 38}:
            return k_position == 23
        elif q_position in {1}:
            return k_position == 33
        elif q_position in {2}:
            return k_position == 20
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {25, 27, 4, 5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8, 9}:
            return k_position == 5
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {19, 11, 12}:
            return k_position == 9
        elif q_position in {13, 14, 15, 21, 26, 28}:
            return k_position == 11
        elif q_position in {16, 17, 22, 23, 24, 29}:
            return k_position == 10
        elif q_position in {18, 58, 42, 37}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 15
        elif q_position in {57, 53, 30}:
            return k_position == 19
        elif q_position in {39, 46, 31}:
            return k_position == 44
        elif q_position in {33, 44}:
            return k_position == 29
        elif q_position in {34}:
            return k_position == 45
        elif q_position in {59, 35}:
            return k_position == 7
        elif q_position in {40, 41, 36}:
            return k_position == 40
        elif q_position in {43}:
            return k_position == 34
        elif q_position in {45}:
            return k_position == 42
        elif q_position in {49, 47}:
            return k_position == 27
        elif q_position in {48}:
            return k_position == 31
        elif q_position in {50}:
            return k_position == 38
        elif q_position in {51}:
            return k_position == 25
        elif q_position in {52}:
            return k_position == 37
        elif q_position in {54}:
            return k_position == 46
        elif q_position in {55}:
            return k_position == 50
        elif q_position in {56}:
            return k_position == 21

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 5, 39, 55, 25, 31}:
            return k_position == 44
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {3, 4, 29}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {49, 7}:
            return k_position == 57
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 42
        elif q_position in {10, 42}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 30
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 32
        elif q_position in {19, 14}:
            return k_position == 10
        elif q_position in {35, 44, 15}:
            return k_position == 29
        elif q_position in {16, 17}:
            return k_position == 15
        elif q_position in {38, 18, 20, 22, 58}:
            return k_position == 7
        elif q_position in {21}:
            return k_position == 13
        elif q_position in {23}:
            return k_position == 17
        elif q_position in {24, 40, 28}:
            return k_position == 21
        elif q_position in {26}:
            return k_position == 23
        elif q_position in {27, 54}:
            return k_position == 34
        elif q_position in {41, 43, 30}:
            return k_position == 48
        elif q_position in {32, 34}:
            return k_position == 58
        elif q_position in {33, 53}:
            return k_position == 50
        elif q_position in {36}:
            return k_position == 40
        elif q_position in {37}:
            return k_position == 38
        elif q_position in {57, 45}:
            return k_position == 52
        elif q_position in {46}:
            return k_position == 56
        elif q_position in {47}:
            return k_position == 25
        elif q_position in {48}:
            return k_position == 47
        elif q_position in {50}:
            return k_position == 35
        elif q_position in {51}:
            return k_position == 36
        elif q_position in {52}:
            return k_position == 53
        elif q_position in {56}:
            return k_position == 49
        elif q_position in {59}:
            return k_position == 46

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 46
        elif q_position in {2}:
            return k_position == 25
        elif q_position in {3, 39}:
            return k_position == 20
        elif q_position in {4}:
            return k_position == 54
        elif q_position in {5}:
            return k_position == 16
        elif q_position in {6}:
            return k_position == 51
        elif q_position in {7}:
            return k_position == 34
        elif q_position in {8, 55}:
            return k_position == 28
        elif q_position in {9, 35, 12, 33}:
            return k_position == 24
        elif q_position in {10}:
            return k_position == 31
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {29, 13, 22}:
            return k_position == 39
        elif q_position in {14}:
            return k_position == 37
        elif q_position in {15}:
            return k_position == 55
        elif q_position in {16}:
            return k_position == 18
        elif q_position in {17, 27}:
            return k_position == 29
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {32, 42, 19, 52, 56, 58}:
            return k_position == 23
        elif q_position in {24, 20}:
            return k_position == 57
        elif q_position in {21}:
            return k_position == 52
        elif q_position in {30, 23}:
            return k_position == 26
        elif q_position in {25}:
            return k_position == 47
        elif q_position in {26, 28}:
            return k_position == 56
        elif q_position in {37, 41, 43, 45, 46, 47, 49, 51, 54, 57, 59, 31}:
            return k_position == 2
        elif q_position in {34}:
            return k_position == 38
        elif q_position in {36}:
            return k_position == 44
        elif q_position in {38}:
            return k_position == 14
        elif q_position in {40}:
            return k_position == 53
        elif q_position in {44}:
            return k_position == 22
        elif q_position in {48}:
            return k_position == 5
        elif q_position in {50}:
            return k_position == 11
        elif q_position in {53}:
            return k_position == 59

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {
            0,
            2,
            4,
            6,
            8,
            10,
            11,
            16,
            18,
            20,
            22,
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
        elif position in {1, 3, 5, 7, 9}:
            return token == ")"
        elif position in {12, 13, 14, 15, 17, 19, 21, 23}:
            return token == "<s>"
        elif position in {42}:
            return token == "("

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 51
        elif token in {")"}:
            return position == 27
        elif token in {"<s>"}:
            return position == 10

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 26
        elif token in {")"}:
            return position == 44
        elif token in {"<s>"}:
            return position == 0

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {
            0,
            9,
            11,
            12,
            13,
            15,
            17,
            18,
            19,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            31,
            32,
            34,
            35,
            36,
            37,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            47,
            48,
            50,
            51,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif position in {16, 1, 20, 14}:
            return token == "<s>"
        elif position in {33, 2, 3, 4, 5, 6, 7, 8, 10, 46, 52, 30}:
            return token == ")"
        elif position in {49, 38}:
            return token == "<pad>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"("}:
            return position == 35
        elif token in {")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 41

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"("}:
            return position == 28
        elif token in {")"}:
            return position == 47
        elif token in {"<s>"}:
            return position == 0

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"("}:
            return position == 8
        elif token in {")"}:
            return position == 30
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {("(", "("), ("(", ")")}:
            return 15
        elif key in {(")", "<s>")}:
            return 2
        elif key in {("(", "<s>"), ("<s>", "("), ("<s>", "<s>")}:
            return 1
        return 24

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, token):
        key = (position, token)
        if key in {
            (0, "("),
            (1, "("),
            (2, "("),
            (3, "("),
            (4, "("),
            (5, "("),
            (6, "("),
            (7, "("),
            (7, ")"),
            (7, "<s>"),
            (8, "("),
            (9, "("),
            (9, ")"),
            (9, "<s>"),
            (10, "("),
            (11, "("),
            (11, ")"),
            (11, "<s>"),
            (12, "("),
            (13, "("),
            (14, "("),
            (15, "("),
            (15, ")"),
            (15, "<s>"),
            (16, "("),
            (17, "("),
            (17, ")"),
            (17, "<s>"),
            (18, "("),
            (19, "("),
            (19, ")"),
            (19, "<s>"),
            (20, "("),
            (21, "("),
            (21, ")"),
            (21, "<s>"),
            (22, "("),
            (23, "("),
            (23, ")"),
            (23, "<s>"),
            (24, "("),
            (25, "("),
            (25, ")"),
            (25, "<s>"),
            (26, "("),
            (27, "("),
            (27, ")"),
            (27, "<s>"),
            (28, "("),
            (29, "("),
            (29, ")"),
            (29, "<s>"),
            (30, "("),
            (31, "("),
            (32, "("),
            (33, "("),
            (34, "("),
            (35, "("),
            (36, "("),
            (37, "("),
            (38, "("),
            (39, "("),
            (40, "("),
            (41, "("),
            (42, "("),
            (43, "("),
            (44, "("),
            (45, "("),
            (46, "("),
            (47, "("),
            (48, "("),
            (49, "("),
            (50, "("),
            (51, "("),
            (52, "("),
            (53, "("),
            (54, "("),
            (55, "("),
            (56, "("),
            (57, "("),
            (58, "("),
            (59, "("),
        }:
            return 22
        elif key in {(5, ")"), (5, "<s>")}:
            return 48
        return 38

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_6_output):
        key = (num_attn_0_0_output, num_attn_0_6_output)
        return 49

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_5_output, num_attn_0_0_output):
        key = (num_attn_0_5_output, num_attn_0_0_output)
        if key in {
            (22, 0),
            (23, 0),
            (24, 0),
            (25, 0),
            (25, 1),
            (26, 0),
            (26, 1),
            (27, 0),
            (27, 1),
            (27, 2),
            (28, 0),
            (28, 1),
            (28, 2),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
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
        }:
            return 55
        return 2

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_1_output, attn_0_0_output):
        if mlp_0_1_output in {
            0,
            1,
            3,
            4,
            5,
            7,
            9,
            11,
            12,
            13,
            19,
            24,
            26,
            28,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
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
            50,
            51,
            52,
            53,
            54,
            56,
            57,
            58,
        }:
            return attn_0_0_output == ""
        elif mlp_0_1_output in {2, 37, 6, 38, 8, 15}:
            return attn_0_0_output == ")"
        elif mlp_0_1_output in {
            59,
            10,
            14,
            16,
            17,
            18,
            49,
            20,
            21,
            22,
            23,
            55,
            25,
            27,
            29,
        }:
            return attn_0_0_output == "("

    attn_1_0_pattern = select_closest(attn_0_0_outputs, mlp_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 44
        elif token in {"<s>"}:
            return position == 5

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_7_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_3_output, position):
        if attn_0_3_output in {"<s>", "("}:
            return position == 1
        elif attn_0_3_output in {")"}:
            return position == 7

    attn_1_2_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_6_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, mlp_0_0_output):
        if token in {")", "("}:
            return mlp_0_0_output == 2
        elif token in {"<s>"}:
            return mlp_0_0_output == 52

    attn_1_3_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_0_output, position):
        if attn_0_0_output in {"<s>", "("}:
            return position == 1
        elif attn_0_0_output in {")"}:
            return position == 5

    attn_1_4_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_4_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_6_output, position):
        if attn_0_6_output in {"("}:
            return position == 3
        elif attn_0_6_output in {")"}:
            return position == 11
        elif attn_0_6_output in {"<s>"}:
            return position == 1

    attn_1_5_pattern = select_closest(positions, attn_0_6_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_1_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 50

    attn_1_6_pattern = select_closest(positions, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_0_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, position):
        if token in {"<s>", "("}:
            return position == 1
        elif token in {")"}:
            return position == 3

    attn_1_7_pattern = select_closest(positions, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_3_output, token):
        if attn_0_3_output in {"<s>", ")", "("}:
            return token == ""

    num_attn_1_0_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_1_output, num_mlp_0_0_output):
        if mlp_0_1_output in {0, 5, 6, 13, 53}:
            return num_mlp_0_0_output == 22
        elif mlp_0_1_output in {1}:
            return num_mlp_0_0_output == 18
        elif mlp_0_1_output in {2}:
            return num_mlp_0_0_output == 52
        elif mlp_0_1_output in {26, 3, 4}:
            return num_mlp_0_0_output == 40
        elif mlp_0_1_output in {8, 7}:
            return num_mlp_0_0_output == 23
        elif mlp_0_1_output in {9, 43}:
            return num_mlp_0_0_output == 20
        elif mlp_0_1_output in {17, 10}:
            return num_mlp_0_0_output == 45
        elif mlp_0_1_output in {40, 11, 21}:
            return num_mlp_0_0_output == 49
        elif mlp_0_1_output in {12, 20, 46}:
            return num_mlp_0_0_output == 15
        elif mlp_0_1_output in {50, 14}:
            return num_mlp_0_0_output == 44
        elif mlp_0_1_output in {18, 15}:
            return num_mlp_0_0_output == 48
        elif mlp_0_1_output in {16, 29}:
            return num_mlp_0_0_output == 14
        elif mlp_0_1_output in {19}:
            return num_mlp_0_0_output == 10
        elif mlp_0_1_output in {22}:
            return num_mlp_0_0_output == 7
        elif mlp_0_1_output in {23}:
            return num_mlp_0_0_output == 12
        elif mlp_0_1_output in {24, 48}:
            return num_mlp_0_0_output == 31
        elif mlp_0_1_output in {25}:
            return num_mlp_0_0_output == 50
        elif mlp_0_1_output in {27}:
            return num_mlp_0_0_output == 58
        elif mlp_0_1_output in {28}:
            return num_mlp_0_0_output == 42
        elif mlp_0_1_output in {30}:
            return num_mlp_0_0_output == 8
        elif mlp_0_1_output in {39, 55, 37, 31}:
            return num_mlp_0_0_output == 34
        elif mlp_0_1_output in {32}:
            return num_mlp_0_0_output == 24
        elif mlp_0_1_output in {33}:
            return num_mlp_0_0_output == 54
        elif mlp_0_1_output in {34}:
            return num_mlp_0_0_output == 25
        elif mlp_0_1_output in {35}:
            return num_mlp_0_0_output == 38
        elif mlp_0_1_output in {36}:
            return num_mlp_0_0_output == 21
        elif mlp_0_1_output in {51, 38}:
            return num_mlp_0_0_output == 36
        elif mlp_0_1_output in {41}:
            return num_mlp_0_0_output == 53
        elif mlp_0_1_output in {42, 44, 45}:
            return num_mlp_0_0_output == 56
        elif mlp_0_1_output in {47}:
            return num_mlp_0_0_output == 51
        elif mlp_0_1_output in {49, 58}:
            return num_mlp_0_0_output == 16
        elif mlp_0_1_output in {52}:
            return num_mlp_0_0_output == 55
        elif mlp_0_1_output in {54}:
            return num_mlp_0_0_output == 35
        elif mlp_0_1_output in {56, 57}:
            return num_mlp_0_0_output == 6
        elif mlp_0_1_output in {59}:
            return num_mlp_0_0_output == 27

    num_attn_1_1_pattern = select(
        num_mlp_0_0_outputs, mlp_0_1_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_7_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, mlp_0_0_output):
        if attn_0_0_output in {"<s>", ")", "("}:
            return mlp_0_0_output == 2

    num_attn_1_2_pattern = select(mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_1_output, position):
        if num_mlp_0_1_output in {0, 1, 53, 21}:
            return position == 23
        elif num_mlp_0_1_output in {33, 2, 7, 8, 9, 43, 12, 46, 52, 22, 24, 25, 29}:
            return position == 20
        elif num_mlp_0_1_output in {3, 36, 6, 48, 18, 51, 23}:
            return position == 24
        elif num_mlp_0_1_output in {34, 4, 11, 50, 19, 55, 26, 28}:
            return position == 25
        elif num_mlp_0_1_output in {37, 5, 10, 44, 45, 47, 49, 58, 59, 31}:
            return position == 18
        elif num_mlp_0_1_output in {56, 41, 13}:
            return position == 19
        elif num_mlp_0_1_output in {32, 38, 14, 54, 57, 27}:
            return position == 22
        elif num_mlp_0_1_output in {15}:
            return position == 40
        elif num_mlp_0_1_output in {16, 17, 42, 20}:
            return position == 21
        elif num_mlp_0_1_output in {30, 39}:
            return position == 26
        elif num_mlp_0_1_output in {35}:
            return position == 27
        elif num_mlp_0_1_output in {40}:
            return position == 16

    num_attn_1_3_pattern = select(positions, num_mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_7_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_6_output, attn_0_5_output):
        if attn_0_6_output in {"<s>", ")", "("}:
            return attn_0_5_output == ""

    num_attn_1_4_pattern = select(attn_0_5_outputs, attn_0_6_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_6_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"<s>", ")", "("}:
            return attn_0_5_output == ""

    num_attn_1_5_pattern = select(attn_0_5_outputs, attn_0_1_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_2_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(mlp_0_1_output, mlp_0_0_output):
        if mlp_0_1_output in {0, 35}:
            return mlp_0_0_output == 20
        elif mlp_0_1_output in {1, 2, 33, 15, 22}:
            return mlp_0_0_output == 45
        elif mlp_0_1_output in {3}:
            return mlp_0_0_output == 29
        elif mlp_0_1_output in {4, 12, 55}:
            return mlp_0_0_output == 57
        elif mlp_0_1_output in {25, 21, 5}:
            return mlp_0_0_output == 44
        elif mlp_0_1_output in {6}:
            return mlp_0_0_output == 47
        elif mlp_0_1_output in {54, 7}:
            return mlp_0_0_output == 21
        elif mlp_0_1_output in {8, 36}:
            return mlp_0_0_output == 49
        elif mlp_0_1_output in {9}:
            return mlp_0_0_output == 22
        elif mlp_0_1_output in {10, 18}:
            return mlp_0_0_output == 5
        elif mlp_0_1_output in {11}:
            return mlp_0_0_output == 52
        elif mlp_0_1_output in {13}:
            return mlp_0_0_output == 10
        elif mlp_0_1_output in {14}:
            return mlp_0_0_output == 31
        elif mlp_0_1_output in {16}:
            return mlp_0_0_output == 53
        elif mlp_0_1_output in {17, 27, 29, 23}:
            return mlp_0_0_output == 26
        elif mlp_0_1_output in {19, 20}:
            return mlp_0_0_output == 25
        elif mlp_0_1_output in {24}:
            return mlp_0_0_output == 48
        elif mlp_0_1_output in {26, 38}:
            return mlp_0_0_output == 15
        elif mlp_0_1_output in {28, 46}:
            return mlp_0_0_output == 58
        elif mlp_0_1_output in {30}:
            return mlp_0_0_output == 46
        elif mlp_0_1_output in {31}:
            return mlp_0_0_output == 39
        elif mlp_0_1_output in {32, 42}:
            return mlp_0_0_output == 42
        elif mlp_0_1_output in {34}:
            return mlp_0_0_output == 59
        elif mlp_0_1_output in {37}:
            return mlp_0_0_output == 12
        elif mlp_0_1_output in {39}:
            return mlp_0_0_output == 54
        elif mlp_0_1_output in {40, 56}:
            return mlp_0_0_output == 33
        elif mlp_0_1_output in {41}:
            return mlp_0_0_output == 56
        elif mlp_0_1_output in {43}:
            return mlp_0_0_output == 4
        elif mlp_0_1_output in {50, 44}:
            return mlp_0_0_output == 27
        elif mlp_0_1_output in {45}:
            return mlp_0_0_output == 14
        elif mlp_0_1_output in {47}:
            return mlp_0_0_output == 17
        elif mlp_0_1_output in {48}:
            return mlp_0_0_output == 37
        elif mlp_0_1_output in {49}:
            return mlp_0_0_output == 41
        elif mlp_0_1_output in {51}:
            return mlp_0_0_output == 55
        elif mlp_0_1_output in {52}:
            return mlp_0_0_output == 23
        elif mlp_0_1_output in {53}:
            return mlp_0_0_output == 0
        elif mlp_0_1_output in {57}:
            return mlp_0_0_output == 7
        elif mlp_0_1_output in {58}:
            return mlp_0_0_output == 34
        elif mlp_0_1_output in {59}:
            return mlp_0_0_output == 36

    num_attn_1_6_pattern = select(mlp_0_0_outputs, mlp_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_7_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_4_output, mlp_0_0_output):
        if attn_0_4_output in {"("}:
            return mlp_0_0_output == 24
        elif attn_0_4_output in {")"}:
            return mlp_0_0_output == 26
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_0_output == 44

    num_attn_1_7_pattern = select(mlp_0_0_outputs, attn_0_4_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_5_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_1_output, attn_0_2_output):
        key = (attn_0_1_output, attn_0_2_output)
        if key in {("(", "("), ("<s>", "(")}:
            return 44
        return 39

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_5_output, attn_0_1_output):
        key = (attn_0_5_output, attn_0_1_output)
        if key in {("(", "(")}:
            return 0
        return 27

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_1_1_output):
        key = (num_attn_1_0_output, num_attn_1_1_output)
        return 40

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output):
        key = num_attn_1_0_output
        return 30

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_1_0_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 3
        elif attn_0_5_output in {")"}:
            return position == 8
        elif attn_0_5_output in {"<s>"}:
            return position == 5

    attn_2_0_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_5_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"("}:
            return position == 11
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 5

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_6_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, mlp_0_0_output):
        if token in {")", "("}:
            return mlp_0_0_output == 2
        elif token in {"<s>"}:
            return mlp_0_0_output == 29

    attn_2_2_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 3
        elif attn_0_4_output in {")"}:
            return position == 5
        elif attn_0_4_output in {"<s>"}:
            return position == 4

    attn_2_3_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 3
        elif attn_0_0_output in {")"}:
            return position == 12
        elif attn_0_0_output in {"<s>"}:
            return position == 1

    attn_2_4_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(token, position):
        if token in {"("}:
            return position == 4
        elif token in {")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 1

    attn_2_5_pattern = select_closest(positions, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_3_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(mlp_0_0_output, position):
        if mlp_0_0_output in {
            0,
            2,
            4,
            5,
            6,
            7,
            8,
            12,
            14,
            16,
            17,
            19,
            21,
            22,
            27,
            32,
            33,
            35,
            37,
            39,
            41,
            49,
            51,
            56,
            57,
            59,
        }:
            return position == 5
        elif mlp_0_0_output in {1, 15}:
            return position == 2
        elif mlp_0_0_output in {3, 10, 45, 50, 54}:
            return position == 7
        elif mlp_0_0_output in {9}:
            return position == 41
        elif mlp_0_0_output in {11}:
            return position == 36
        elif mlp_0_0_output in {13, 30, 47}:
            return position == 45
        elif mlp_0_0_output in {25, 18, 20}:
            return position == 4
        elif mlp_0_0_output in {58, 23}:
            return position == 6
        elif mlp_0_0_output in {24, 34}:
            return position == 10
        elif mlp_0_0_output in {40, 43, 44, 48, 26}:
            return position == 9
        elif mlp_0_0_output in {28}:
            return position == 12
        elif mlp_0_0_output in {29}:
            return position == 11
        elif mlp_0_0_output in {31}:
            return position == 43
        elif mlp_0_0_output in {36, 46}:
            return position == 30
        elif mlp_0_0_output in {38}:
            return position == 35
        elif mlp_0_0_output in {42}:
            return position == 37
        elif mlp_0_0_output in {52}:
            return position == 1
        elif mlp_0_0_output in {53}:
            return position == 40
        elif mlp_0_0_output in {55}:
            return position == 50

    attn_2_6_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_1_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(token, position):
        if token in {"("}:
            return position == 3
        elif token in {")"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 5

    attn_2_7_pattern = select_closest(positions, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_1_0_output, mlp_1_1_output):
        if mlp_1_0_output in {
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
            12,
            13,
            16,
            17,
            19,
            23,
            24,
            25,
            28,
            29,
            30,
            31,
            32,
            33,
            35,
            36,
            37,
            39,
            40,
            41,
            43,
            44,
            45,
            46,
            48,
            49,
            50,
            51,
            54,
            55,
            57,
            58,
            59,
        }:
            return mlp_1_1_output == 27
        elif mlp_1_0_output in {10, 26, 52}:
            return mlp_1_1_output == 6
        elif mlp_1_0_output in {11}:
            return mlp_1_1_output == 15
        elif mlp_1_0_output in {14}:
            return mlp_1_1_output == 39
        elif mlp_1_0_output in {34, 42, 15, 18, 20, 22}:
            return mlp_1_1_output == 16
        elif mlp_1_0_output in {21}:
            return mlp_1_1_output == 24
        elif mlp_1_0_output in {27}:
            return mlp_1_1_output == 40
        elif mlp_1_0_output in {38}:
            return mlp_1_1_output == 0
        elif mlp_1_0_output in {47}:
            return mlp_1_1_output == 44
        elif mlp_1_0_output in {53}:
            return mlp_1_1_output == 26
        elif mlp_1_0_output in {56}:
            return mlp_1_1_output == 53

    num_attn_2_0_pattern = select(mlp_1_1_outputs, mlp_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_1_0_output, attn_1_3_output):
        if num_mlp_1_0_output in {
            0,
            1,
            2,
            3,
            4,
            5,
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
            22,
            24,
            25,
            26,
            27,
            28,
            29,
            42,
            43,
            44,
            45,
            46,
            47,
            52,
            57,
            59,
        }:
            return attn_1_3_output == ""
        elif num_mlp_1_0_output in {
            32,
            34,
            35,
            36,
            37,
            7,
            40,
            41,
            48,
            49,
            50,
            51,
            53,
            54,
            55,
            58,
            30,
            31,
        }:
            return attn_1_3_output == "<s>"
        elif num_mlp_1_0_output in {56, 33, 21, 23}:
            return attn_1_3_output == "<pad>"
        elif num_mlp_1_0_output in {38}:
            return attn_1_3_output == ")"
        elif num_mlp_1_0_output in {39}:
            return attn_1_3_output == "("

    num_attn_2_1_pattern = select(
        attn_1_3_outputs, num_mlp_1_0_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_7_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_4_output, attn_1_4_output):
        if attn_0_4_output in {"<s>", ")", "("}:
            return attn_1_4_output == ""

    num_attn_2_2_pattern = select(attn_1_4_outputs, attn_0_4_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_0_output, num_mlp_0_1_output):
        if mlp_0_0_output in {0}:
            return num_mlp_0_1_output == 7
        elif mlp_0_0_output in {1, 26, 28, 14}:
            return num_mlp_0_1_output == 16
        elif mlp_0_0_output in {2, 54}:
            return num_mlp_0_1_output == 41
        elif mlp_0_0_output in {35, 3, 4}:
            return num_mlp_0_1_output == 43
        elif mlp_0_0_output in {12, 5}:
            return num_mlp_0_1_output == 9
        elif mlp_0_0_output in {6}:
            return num_mlp_0_1_output == 10
        elif mlp_0_0_output in {7}:
            return num_mlp_0_1_output == 18
        elif mlp_0_0_output in {8, 32}:
            return num_mlp_0_1_output == 6
        elif mlp_0_0_output in {9, 53, 30, 31}:
            return num_mlp_0_1_output == 0
        elif mlp_0_0_output in {10, 43, 21}:
            return num_mlp_0_1_output == 35
        elif mlp_0_0_output in {11}:
            return num_mlp_0_1_output == 29
        elif mlp_0_0_output in {40, 41, 13}:
            return num_mlp_0_1_output == 59
        elif mlp_0_0_output in {29, 15}:
            return num_mlp_0_1_output == 13
        elif mlp_0_0_output in {16, 56}:
            return num_mlp_0_1_output == 50
        elif mlp_0_0_output in {17, 25}:
            return num_mlp_0_1_output == 55
        elif mlp_0_0_output in {18, 50, 55, 47}:
            return num_mlp_0_1_output == 49
        elif mlp_0_0_output in {19}:
            return num_mlp_0_1_output == 1
        elif mlp_0_0_output in {20}:
            return num_mlp_0_1_output == 8
        elif mlp_0_0_output in {24, 48, 27, 22}:
            return num_mlp_0_1_output == 23
        elif mlp_0_0_output in {46, 23}:
            return num_mlp_0_1_output == 36
        elif mlp_0_0_output in {33}:
            return num_mlp_0_1_output == 19
        elif mlp_0_0_output in {34, 58, 38}:
            return num_mlp_0_1_output == 37
        elif mlp_0_0_output in {36}:
            return num_mlp_0_1_output == 58
        elif mlp_0_0_output in {37, 39}:
            return num_mlp_0_1_output == 25
        elif mlp_0_0_output in {42}:
            return num_mlp_0_1_output == 20
        elif mlp_0_0_output in {44}:
            return num_mlp_0_1_output == 54
        elif mlp_0_0_output in {45}:
            return num_mlp_0_1_output == 5
        elif mlp_0_0_output in {49}:
            return num_mlp_0_1_output == 24
        elif mlp_0_0_output in {51}:
            return num_mlp_0_1_output == 44
        elif mlp_0_0_output in {52}:
            return num_mlp_0_1_output == 53
        elif mlp_0_0_output in {57}:
            return num_mlp_0_1_output == 40
        elif mlp_0_0_output in {59}:
            return num_mlp_0_1_output == 17

    num_attn_2_3_pattern = select(
        num_mlp_0_1_outputs, mlp_0_0_outputs, num_predicate_2_3
    )
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(mlp_1_0_output, attn_0_5_output):
        if mlp_1_0_output in {
            0,
            1,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            13,
            14,
            15,
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
            46,
            47,
            48,
            49,
            51,
            52,
            53,
            55,
            56,
            57,
            58,
            59,
        }:
            return attn_0_5_output == ""
        elif mlp_1_0_output in {32, 33, 2, 3, 39, 11, 12, 45, 17, 18, 19, 20, 50, 54}:
            return attn_0_5_output == "("
        elif mlp_1_0_output in {21}:
            return attn_0_5_output == "<s>"

    num_attn_2_4_pattern = select(attn_0_5_outputs, mlp_1_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_7_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(mlp_1_0_output, mlp_1_1_output):
        if mlp_1_0_output in {0, 34, 15, 48, 50, 24}:
            return mlp_1_1_output == 20
        elif mlp_1_0_output in {1, 35, 4, 39, 40, 20, 56, 25, 29}:
            return mlp_1_1_output == 16
        elif mlp_1_0_output in {2, 45}:
            return mlp_1_1_output == 35
        elif mlp_1_0_output in {3}:
            return mlp_1_1_output == 30
        elif mlp_1_0_output in {37, 5, 43, 44, 52, 23, 58, 30}:
            return mlp_1_1_output == 0
        elif mlp_1_0_output in {17, 6}:
            return mlp_1_1_output == 50
        elif mlp_1_0_output in {8, 7}:
            return mlp_1_1_output == 44
        elif mlp_1_0_output in {9}:
            return mlp_1_1_output == 51
        elif mlp_1_0_output in {10}:
            return mlp_1_1_output == 29
        elif mlp_1_0_output in {11}:
            return mlp_1_1_output == 6
        elif mlp_1_0_output in {19, 12}:
            return mlp_1_1_output == 42
        elif mlp_1_0_output in {13}:
            return mlp_1_1_output == 19
        elif mlp_1_0_output in {14}:
            return mlp_1_1_output == 12
        elif mlp_1_0_output in {16}:
            return mlp_1_1_output == 36
        elif mlp_1_0_output in {18}:
            return mlp_1_1_output == 38
        elif mlp_1_0_output in {21}:
            return mlp_1_1_output == 56
        elif mlp_1_0_output in {22}:
            return mlp_1_1_output == 43
        elif mlp_1_0_output in {26}:
            return mlp_1_1_output == 40
        elif mlp_1_0_output in {42, 27, 36}:
            return mlp_1_1_output == 31
        elif mlp_1_0_output in {28}:
            return mlp_1_1_output == 25
        elif mlp_1_0_output in {33, 31}:
            return mlp_1_1_output == 26
        elif mlp_1_0_output in {32}:
            return mlp_1_1_output == 24
        elif mlp_1_0_output in {57, 38}:
            return mlp_1_1_output == 45
        elif mlp_1_0_output in {41}:
            return mlp_1_1_output == 34
        elif mlp_1_0_output in {46}:
            return mlp_1_1_output == 22
        elif mlp_1_0_output in {47}:
            return mlp_1_1_output == 47
        elif mlp_1_0_output in {49}:
            return mlp_1_1_output == 17
        elif mlp_1_0_output in {51}:
            return mlp_1_1_output == 15
        elif mlp_1_0_output in {53, 55}:
            return mlp_1_1_output == 23
        elif mlp_1_0_output in {54}:
            return mlp_1_1_output == 21
        elif mlp_1_0_output in {59}:
            return mlp_1_1_output == 41

    num_attn_2_5_pattern = select(mlp_1_1_outputs, mlp_1_0_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_7_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_1_output, attn_1_3_output):
        if attn_1_1_output in {"<s>", ")", "("}:
            return attn_1_3_output == ""

    num_attn_2_6_pattern = select(attn_1_3_outputs, attn_1_1_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_6_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(mlp_1_1_output, mlp_0_0_output):
        if mlp_1_1_output in {0}:
            return mlp_0_0_output == 36
        elif mlp_1_1_output in {46, 1, 30}:
            return mlp_0_0_output == 6
        elif mlp_1_1_output in {17, 2, 11}:
            return mlp_0_0_output == 55
        elif mlp_1_1_output in {3, 41, 19, 25, 29}:
            return mlp_0_0_output == 15
        elif mlp_1_1_output in {4}:
            return mlp_0_0_output == 49
        elif mlp_1_1_output in {16, 56, 58, 5}:
            return mlp_0_0_output == 32
        elif mlp_1_1_output in {21, 6}:
            return mlp_0_0_output == 53
        elif mlp_1_1_output in {18, 39, 7}:
            return mlp_0_0_output == 45
        elif mlp_1_1_output in {8, 50}:
            return mlp_0_0_output == 57
        elif mlp_1_1_output in {9, 10, 35, 55}:
            return mlp_0_0_output == 58
        elif mlp_1_1_output in {40, 12, 14, 22, 28}:
            return mlp_0_0_output == 2
        elif mlp_1_1_output in {48, 13}:
            return mlp_0_0_output == 48
        elif mlp_1_1_output in {51, 15}:
            return mlp_0_0_output == 23
        elif mlp_1_1_output in {20}:
            return mlp_0_0_output == 12
        elif mlp_1_1_output in {23}:
            return mlp_0_0_output == 16
        elif mlp_1_1_output in {24}:
            return mlp_0_0_output == 21
        elif mlp_1_1_output in {26, 34}:
            return mlp_0_0_output == 19
        elif mlp_1_1_output in {27}:
            return mlp_0_0_output == 40
        elif mlp_1_1_output in {31}:
            return mlp_0_0_output == 31
        elif mlp_1_1_output in {32}:
            return mlp_0_0_output == 37
        elif mlp_1_1_output in {33}:
            return mlp_0_0_output == 3
        elif mlp_1_1_output in {36}:
            return mlp_0_0_output == 17
        elif mlp_1_1_output in {37}:
            return mlp_0_0_output == 10
        elif mlp_1_1_output in {49, 38}:
            return mlp_0_0_output == 41
        elif mlp_1_1_output in {42, 53}:
            return mlp_0_0_output == 52
        elif mlp_1_1_output in {43}:
            return mlp_0_0_output == 18
        elif mlp_1_1_output in {44}:
            return mlp_0_0_output == 25
        elif mlp_1_1_output in {45}:
            return mlp_0_0_output == 30
        elif mlp_1_1_output in {59, 47}:
            return mlp_0_0_output == 11
        elif mlp_1_1_output in {52}:
            return mlp_0_0_output == 8
        elif mlp_1_1_output in {57, 54}:
            return mlp_0_0_output == 39

    num_attn_2_7_pattern = select(mlp_0_0_outputs, mlp_1_1_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_2_4_output):
        key = (attn_2_3_output, attn_2_4_output)
        if key in {("(", ")"), ("<s>", ")")}:
            return 30
        elif key in {(")", ")")}:
            return 3
        return 34

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_4_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_3_output):
        key = attn_2_3_output
        if key in {"", ")"}:
            return 48
        elif key in {""}:
            return 21
        return 46

    mlp_2_1_outputs = [mlp_2_1(k0) for k0 in attn_2_3_outputs]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_2_output, num_attn_2_1_output):
        key = (num_attn_2_2_output, num_attn_2_1_output)
        return 49

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_6_output, num_attn_1_0_output):
        key = (num_attn_2_6_output, num_attn_1_0_output)
        return 39

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_6_outputs, num_attn_1_0_outputs)
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
            "(",
            "(",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
            ")",
            "(",
            "(",
            "(",
        ]
    )
)
