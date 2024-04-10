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
        "output/length/rasp/dyck1/trainlength40/s0/dyck1_weights.csv",
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
        if q_position in {0, 71}:
            return k_position == 66
        elif q_position in {1, 59}:
            return k_position == 78
        elif q_position in {2, 13}:
            return k_position == 10
        elif q_position in {3}:
            return k_position == 57
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {9, 5}:
            return k_position == 45
        elif q_position in {37, 6}:
            return k_position == 5
        elif q_position in {75, 7}:
            return k_position == 33
        elif q_position in {35, 39, 8, 10, 18, 26, 28}:
            return k_position == 6
        elif q_position in {11, 76, 23}:
            return k_position == 9
        elif q_position in {12, 14}:
            return k_position == 8
        elif q_position in {20, 36, 15}:
            return k_position == 13
        elif q_position in {16, 17, 38}:
            return k_position == 12
        elif q_position in {19, 79}:
            return k_position == 15
        elif q_position in {25, 21}:
            return k_position == 14
        elif q_position in {51, 22, 63}:
            return k_position == 17
        elif q_position in {24}:
            return k_position == 22
        elif q_position in {34, 48, 27, 29, 30}:
            return k_position == 11
        elif q_position in {69, 70, 31}:
            return k_position == 19
        elif q_position in {32}:
            return k_position == 27
        elif q_position in {33}:
            return k_position == 4
        elif q_position in {40, 46}:
            return k_position == 46
        elif q_position in {41, 47}:
            return k_position == 49
        elif q_position in {42}:
            return k_position == 41
        elif q_position in {43, 53}:
            return k_position == 67
        elif q_position in {64, 44}:
            return k_position == 65
        elif q_position in {45}:
            return k_position == 43
        elif q_position in {49, 50}:
            return k_position == 39
        elif q_position in {52}:
            return k_position == 55
        elif q_position in {56, 54}:
            return k_position == 1
        elif q_position in {55}:
            return k_position == 52
        elif q_position in {57}:
            return k_position == 58
        elif q_position in {58}:
            return k_position == 59
        elif q_position in {60}:
            return k_position == 50
        elif q_position in {66, 61}:
            return k_position == 28
        elif q_position in {72, 62}:
            return k_position == 23
        elif q_position in {65, 68}:
            return k_position == 79
        elif q_position in {67}:
            return k_position == 30
        elif q_position in {73}:
            return k_position == 53
        elif q_position in {74}:
            return k_position == 56
        elif q_position in {77}:
            return k_position == 62
        elif q_position in {78}:
            return k_position == 54

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 57

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 3

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 24, 72, 30}:
            return k_position == 23
        elif q_position in {1, 28}:
            return k_position == 8
        elif q_position in {2, 6}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4, 47}:
            return k_position == 3
        elif q_position in {73, 5}:
            return k_position == 38
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9, 39}:
            return k_position == 6
        elif q_position in {32, 71, 10, 12, 23}:
            return k_position == 9
        elif q_position in {19, 11, 44, 78}:
            return k_position == 10
        elif q_position in {36, 13}:
            return k_position == 12
        elif q_position in {38, 66, 14, 63}:
            return k_position == 13
        elif q_position in {34, 15}:
            return k_position == 14
        elif q_position in {16, 55}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18, 50, 60}:
            return k_position == 17
        elif q_position in {20, 46}:
            return k_position == 19
        elif q_position in {77, 21}:
            return k_position == 18
        elif q_position in {42, 52, 22}:
            return k_position == 21
        elif q_position in {68, 70, 76, 79, 25}:
            return k_position == 22
        elif q_position in {41, 26, 45}:
            return k_position == 25
        elif q_position in {27, 61}:
            return k_position == 26
        elif q_position in {43, 29}:
            return k_position == 28
        elif q_position in {31}:
            return k_position == 30
        elif q_position in {33, 37}:
            return k_position == 32
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {40}:
            return k_position == 31
        elif q_position in {48}:
            return k_position == 74
        elif q_position in {49}:
            return k_position == 59
        elif q_position in {59, 51}:
            return k_position == 35
        elif q_position in {53}:
            return k_position == 24
        elif q_position in {67, 54}:
            return k_position == 77
        elif q_position in {56, 58, 64}:
            return k_position == 29
        elif q_position in {57}:
            return k_position == 42
        elif q_position in {62}:
            return k_position == 71
        elif q_position in {65}:
            return k_position == 43
        elif q_position in {69}:
            return k_position == 56
        elif q_position in {74}:
            return k_position == 20
        elif q_position in {75}:
            return k_position == 76

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 49, 10, 78}:
            return k_position == 9
        elif q_position in {1, 66, 33, 31, 63}:
            return k_position == 21
        elif q_position in {2, 37, 30, 39}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5, 7}:
            return k_position == 78
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {32, 9}:
            return k_position == 6
        elif q_position in {11}:
            return k_position == 38
        elif q_position in {12}:
            return k_position == 8
        elif q_position in {71, 13, 22, 15}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {38, 41, 73, 16, 18, 57}:
            return k_position == 15
        elif q_position in {72, 17, 19, 53, 21}:
            return k_position == 16
        elif q_position in {25, 20, 70}:
            return k_position == 19
        elif q_position in {23}:
            return k_position == 12
        elif q_position in {24, 29, 55}:
            return k_position == 22
        elif q_position in {26, 59}:
            return k_position == 23
        elif q_position in {27, 28}:
            return k_position == 20
        elif q_position in {34}:
            return k_position == 27
        elif q_position in {35}:
            return k_position == 26
        elif q_position in {36}:
            return k_position == 29
        elif q_position in {40}:
            return k_position == 28
        elif q_position in {42}:
            return k_position == 31
        elif q_position in {43}:
            return k_position == 44
        elif q_position in {44}:
            return k_position == 36
        elif q_position in {67, 60, 45}:
            return k_position == 55
        elif q_position in {46}:
            return k_position == 58
        elif q_position in {47}:
            return k_position == 53
        elif q_position in {48}:
            return k_position == 45
        elif q_position in {50}:
            return k_position == 48
        elif q_position in {65, 51}:
            return k_position == 72
        elif q_position in {52}:
            return k_position == 65
        elif q_position in {77, 54}:
            return k_position == 30
        elif q_position in {56}:
            return k_position == 49
        elif q_position in {58}:
            return k_position == 17
        elif q_position in {61}:
            return k_position == 69
        elif q_position in {62}:
            return k_position == 42
        elif q_position in {64}:
            return k_position == 33
        elif q_position in {68}:
            return k_position == 39
        elif q_position in {69}:
            return k_position == 79
        elif q_position in {74}:
            return k_position == 25
        elif q_position in {75}:
            return k_position == 41
        elif q_position in {76}:
            return k_position == 59
        elif q_position in {79}:
            return k_position == 43

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 3

    attn_0_5_pattern = select_closest(positions, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 73

    attn_0_6_pattern = select_closest(positions, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 48, 67}:
            return k_position == 32
        elif q_position in {1}:
            return k_position == 75
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4, 5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {45, 7}:
            return k_position == 62
        elif q_position in {33, 35, 37, 39, 8, 11, 12, 14, 20, 23, 24, 26, 29, 31}:
            return k_position == 6
        elif q_position in {9, 73, 46, 56, 62}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {36, 13, 51, 19, 61}:
            return k_position == 11
        elif q_position in {32, 15, 16, 17, 18, 47, 21, 30}:
            return k_position == 10
        elif q_position in {25, 28, 22}:
            return k_position == 9
        elif q_position in {27}:
            return k_position == 17
        elif q_position in {34}:
            return k_position == 12
        elif q_position in {53, 38, 55}:
            return k_position == 33
        elif q_position in {40, 41, 54}:
            return k_position == 31
        elif q_position in {42}:
            return k_position == 48
        elif q_position in {43}:
            return k_position == 78
        elif q_position in {44}:
            return k_position == 59
        elif q_position in {49, 59, 78, 63}:
            return k_position == 5
        elif q_position in {64, 50}:
            return k_position == 39
        elif q_position in {74, 52}:
            return k_position == 56
        elif q_position in {57}:
            return k_position == 60
        elif q_position in {58}:
            return k_position == 52
        elif q_position in {60}:
            return k_position == 61
        elif q_position in {65}:
            return k_position == 53
        elif q_position in {66}:
            return k_position == 34
        elif q_position in {68}:
            return k_position == 71
        elif q_position in {69}:
            return k_position == 13
        elif q_position in {70}:
            return k_position == 44
        elif q_position in {71}:
            return k_position == 64
        elif q_position in {72}:
            return k_position == 42
        elif q_position in {75}:
            return k_position == 76
        elif q_position in {76}:
            return k_position == 40
        elif q_position in {77}:
            return k_position == 50
        elif q_position in {79}:
            return k_position == 57

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 36
        elif q_position in {16, 1, 6}:
            return k_position == 54
        elif q_position in {2, 45}:
            return k_position == 79
        elif q_position in {17, 3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 26
        elif q_position in {5}:
            return k_position == 29
        elif q_position in {37, 13, 7}:
            return k_position == 23
        elif q_position in {8, 68}:
            return k_position == 37
        elif q_position in {9}:
            return k_position == 31
        elif q_position in {10}:
            return k_position == 56
        elif q_position in {11}:
            return k_position == 24
        elif q_position in {12}:
            return k_position == 5
        elif q_position in {14, 39}:
            return k_position == 72
        elif q_position in {44, 15}:
            return k_position == 73
        elif q_position in {32, 36, 73, 18, 25, 62}:
            return k_position == 51
        elif q_position in {19}:
            return k_position == 49
        elif q_position in {34, 27, 20}:
            return k_position == 42
        elif q_position in {21}:
            return k_position == 30
        elif q_position in {33, 74, 22}:
            return k_position == 66
        elif q_position in {23}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 28
        elif q_position in {26, 60}:
            return k_position == 55
        elif q_position in {28, 46, 52}:
            return k_position == 69
        elif q_position in {57, 29}:
            return k_position == 74
        elif q_position in {76, 30}:
            return k_position == 12
        elif q_position in {64, 47, 78, 31}:
            return k_position == 76
        elif q_position in {35}:
            return k_position == 60
        elif q_position in {38}:
            return k_position == 71
        elif q_position in {40}:
            return k_position == 46
        elif q_position in {41, 71}:
            return k_position == 58
        elif q_position in {42}:
            return k_position == 65
        elif q_position in {56, 43}:
            return k_position == 68
        elif q_position in {48, 70}:
            return k_position == 39
        elif q_position in {49}:
            return k_position == 70
        elif q_position in {50}:
            return k_position == 17
        elif q_position in {51}:
            return k_position == 16
        elif q_position in {53}:
            return k_position == 78
        elif q_position in {54}:
            return k_position == 44
        elif q_position in {58, 55}:
            return k_position == 52
        elif q_position in {59}:
            return k_position == 77
        elif q_position in {61}:
            return k_position == 22
        elif q_position in {75, 63}:
            return k_position == 75
        elif q_position in {72, 65, 77}:
            return k_position == 59
        elif q_position in {66}:
            return k_position == 53
        elif q_position in {67}:
            return k_position == 21
        elif q_position in {69}:
            return k_position == 62
        elif q_position in {79}:
            return k_position == 13

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 76}:
            return k_position == 25
        elif q_position in {1}:
            return k_position == 46
        elif q_position in {2}:
            return k_position == 77
        elif q_position in {3, 15}:
            return k_position == 4
        elif q_position in {41, 4}:
            return k_position == 1
        elif q_position in {28, 5}:
            return k_position == 60
        elif q_position in {6}:
            return k_position == 75
        elif q_position in {8, 35, 30, 7}:
            return k_position == 50
        elif q_position in {9, 26, 20}:
            return k_position == 63
        elif q_position in {10, 27}:
            return k_position == 71
        elif q_position in {11}:
            return k_position == 57
        elif q_position in {12}:
            return k_position == 35
        elif q_position in {13}:
            return k_position == 73
        elif q_position in {14}:
            return k_position == 40
        elif q_position in {16, 36}:
            return k_position == 67
        elif q_position in {32, 17}:
            return k_position == 45
        elif q_position in {18}:
            return k_position == 70
        elif q_position in {19}:
            return k_position == 55
        elif q_position in {56, 51, 21}:
            return k_position == 27
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 72
        elif q_position in {24}:
            return k_position == 42
        elif q_position in {25}:
            return k_position == 36
        elif q_position in {65, 40, 49, 29, 63}:
            return k_position == 30
        elif q_position in {31}:
            return k_position == 48
        elif q_position in {33, 43}:
            return k_position == 34
        elif q_position in {34}:
            return k_position == 69
        elif q_position in {37}:
            return k_position == 58
        elif q_position in {38}:
            return k_position == 47
        elif q_position in {42, 39}:
            return k_position == 64
        elif q_position in {66, 44}:
            return k_position == 29
        elif q_position in {64, 70, 72, 45, 78, 48}:
            return k_position == 2
        elif q_position in {69, 77, 46, 53, 55, 58, 61}:
            return k_position == 28
        elif q_position in {73, 47, 57, 59, 62}:
            return k_position == 32
        elif q_position in {68, 71, 74, 75, 50, 52, 54}:
            return k_position == 31
        elif q_position in {67, 60}:
            return k_position == 33
        elif q_position in {79}:
            return k_position == 26

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 17
        elif token in {")"}:
            return position == 68
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 38
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 12

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            15,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            33,
            34,
            35,
            36,
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
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return token == ""
        elif position in {1, 3, 5, 7, 9}:
            return token == ")"
        elif position in {37, 11, 13, 17, 19, 21, 23, 25, 27, 29, 31}:
            return token == "<s>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"("}:
            return position == 3
        elif token in {")"}:
            return position == 74
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0, 65, 34, 50}:
            return k_position == 50
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 10
        elif q_position in {3, 5, 7}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 21
        elif q_position in {6}:
            return k_position == 16
        elif q_position in {33, 8, 11, 13, 17, 20}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 9
        elif q_position in {10, 68, 14}:
            return k_position == 5
        elif q_position in {16, 64, 12}:
            return k_position == 77
        elif q_position in {72, 51, 15}:
            return k_position == 67
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {19, 37}:
            return k_position == 48
        elif q_position in {56, 77, 21, 46}:
            return k_position == 55
        elif q_position in {22}:
            return k_position == 26
        elif q_position in {23}:
            return k_position == 28
        elif q_position in {24}:
            return k_position == 32
        elif q_position in {25}:
            return k_position == 35
        elif q_position in {26, 39}:
            return k_position == 75
        elif q_position in {27}:
            return k_position == 52
        elif q_position in {28}:
            return k_position == 38
        elif q_position in {29}:
            return k_position == 44
        elif q_position in {30}:
            return k_position == 63
        elif q_position in {40, 43, 31}:
            return k_position == 62
        elif q_position in {32}:
            return k_position == 71
        elif q_position in {35}:
            return k_position == 40
        elif q_position in {36}:
            return k_position == 76
        elif q_position in {58, 38, 63}:
            return k_position == 54
        elif q_position in {41, 54}:
            return k_position == 46
        elif q_position in {42}:
            return k_position == 11
        elif q_position in {44}:
            return k_position == 47
        elif q_position in {45}:
            return k_position == 51
        elif q_position in {47}:
            return k_position == 57
        elif q_position in {48}:
            return k_position == 59
        elif q_position in {49}:
            return k_position == 70
        elif q_position in {52}:
            return k_position == 72
        elif q_position in {53}:
            return k_position == 73
        elif q_position in {69, 55}:
            return k_position == 79
        elif q_position in {57}:
            return k_position == 65
        elif q_position in {59}:
            return k_position == 53
        elif q_position in {60}:
            return k_position == 39
        elif q_position in {61}:
            return k_position == 69
        elif q_position in {62}:
            return k_position == 6
        elif q_position in {66}:
            return k_position == 43
        elif q_position in {67}:
            return k_position == 61
        elif q_position in {70}:
            return k_position == 78
        elif q_position in {71}:
            return k_position == 74
        elif q_position in {73}:
            return k_position == 7
        elif q_position in {74}:
            return k_position == 49
        elif q_position in {75}:
            return k_position == 66
        elif q_position in {76, 78}:
            return k_position == 12
        elif q_position in {79}:
            return k_position == 13

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {
            0,
            1,
            2,
            7,
            9,
            10,
            11,
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
            49,
            50,
            53,
            60,
            61,
            63,
            69,
            71,
            79,
        }:
            return token == ""
        elif position in {
            3,
            4,
            5,
            6,
            8,
            12,
            40,
            44,
            45,
            46,
            48,
            51,
            52,
            54,
            55,
            56,
            57,
            58,
            59,
            62,
            64,
            65,
            66,
            67,
            68,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
        }:
            return token == ")"
        elif position in {24}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_3_output):
        key = (attn_0_1_output, attn_0_3_output)
        if key in {("(", ")"), (")", ")"), ("<s>", ")")}:
            return 65
        elif key in {(")", "<s>"), ("<s>", "<s>")}:
            return 75
        return 17

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_4_output, attn_0_5_output):
        key = (attn_0_4_output, attn_0_5_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 2
        elif key in {("(", "("), ("(", ")"), ("(", "<s>"), ("<s>", "(")}:
            return 17
        return 21

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_5_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output, num_attn_0_5_output):
        key = (num_attn_0_3_output, num_attn_0_5_output)
        if key in {
            (0, 72),
            (0, 73),
            (0, 74),
            (0, 75),
            (0, 76),
            (0, 77),
            (0, 78),
            (0, 79),
        }:
            return 68
        return 0

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_4_output, num_attn_0_0_output):
        key = (num_attn_0_4_output, num_attn_0_0_output)
        return 22

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 67, 4, 8, 11}:
            return position == 22
        elif mlp_0_0_output in {16, 1}:
            return position == 75
        elif mlp_0_0_output in {2}:
            return position == 65
        elif mlp_0_0_output in {3}:
            return position == 77
        elif mlp_0_0_output in {34, 5, 10, 47, 48}:
            return position == 19
        elif mlp_0_0_output in {49, 52, 77, 6}:
            return position == 21
        elif mlp_0_0_output in {69, 70, 7, 46, 62}:
            return position == 23
        elif mlp_0_0_output in {9}:
            return position == 28
        elif mlp_0_0_output in {74, 12, 13, 15, 51, 19}:
            return position == 37
        elif mlp_0_0_output in {59, 28, 21, 14}:
            return position == 33
        elif mlp_0_0_output in {17}:
            return position == 2
        elif mlp_0_0_output in {18}:
            return position == 44
        elif mlp_0_0_output in {20, 78}:
            return position == 26
        elif mlp_0_0_output in {22}:
            return position == 41
        elif mlp_0_0_output in {39, 23}:
            return position == 5
        elif mlp_0_0_output in {24, 40, 64}:
            return position == 13
        elif mlp_0_0_output in {65, 25, 41}:
            return position == 10
        elif mlp_0_0_output in {26, 76}:
            return position == 31
        elif mlp_0_0_output in {42, 27}:
            return position == 8
        elif mlp_0_0_output in {29}:
            return position == 32
        elif mlp_0_0_output in {57, 30}:
            return position == 3
        elif mlp_0_0_output in {31}:
            return position == 6
        elif mlp_0_0_output in {32}:
            return position == 69
        elif mlp_0_0_output in {33}:
            return position == 73
        elif mlp_0_0_output in {35, 36}:
            return position == 9
        elif mlp_0_0_output in {37}:
            return position == 54
        elif mlp_0_0_output in {60, 38}:
            return position == 11
        elif mlp_0_0_output in {43, 61}:
            return position == 35
        elif mlp_0_0_output in {44}:
            return position == 29
        elif mlp_0_0_output in {45}:
            return position == 14
        elif mlp_0_0_output in {72, 73, 50, 79}:
            return position == 15
        elif mlp_0_0_output in {58, 68, 53, 54}:
            return position == 24
        elif mlp_0_0_output in {63, 55}:
            return position == 27
        elif mlp_0_0_output in {56}:
            return position == 55
        elif mlp_0_0_output in {66, 75, 71}:
            return position == 18

    attn_1_0_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_5_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 50, 52, 69}:
            return position == 18
        elif mlp_0_0_output in {1}:
            return position == 57
        elif mlp_0_0_output in {2}:
            return position == 73
        elif mlp_0_0_output in {32, 3, 67, 38, 18, 24}:
            return position == 33
        elif mlp_0_0_output in {4, 70, 76, 46, 48, 55}:
            return position == 24
        elif mlp_0_0_output in {5}:
            return position == 22
        elif mlp_0_0_output in {40, 64, 6}:
            return position == 21
        elif mlp_0_0_output in {73, 7}:
            return position == 15
        elif mlp_0_0_output in {8, 23}:
            return position == 28
        elif mlp_0_0_output in {71, 9, 10, 11, 47, 16, 26}:
            return position == 19
        elif mlp_0_0_output in {74, 43, 12, 13, 19}:
            return position == 37
        elif mlp_0_0_output in {14}:
            return position == 36
        elif mlp_0_0_output in {15}:
            return position == 59
        elif mlp_0_0_output in {17, 49}:
            return position == 3
        elif mlp_0_0_output in {20}:
            return position == 41
        elif mlp_0_0_output in {25, 21}:
            return position == 7
        elif mlp_0_0_output in {22}:
            return position == 45
        elif mlp_0_0_output in {27}:
            return position == 10
        elif mlp_0_0_output in {59, 28}:
            return position == 44
        elif mlp_0_0_output in {29}:
            return position == 23
        elif mlp_0_0_output in {30}:
            return position == 6
        elif mlp_0_0_output in {77, 31}:
            return position == 5
        elif mlp_0_0_output in {33, 42, 41}:
            return position == 8
        elif mlp_0_0_output in {34, 66}:
            return position == 34
        elif mlp_0_0_output in {35}:
            return position == 55
        elif mlp_0_0_output in {72, 57, 36, 54}:
            return position == 13
        elif mlp_0_0_output in {37}:
            return position == 54
        elif mlp_0_0_output in {39}:
            return position == 1
        elif mlp_0_0_output in {44, 68, 63}:
            return position == 17
        elif mlp_0_0_output in {65, 53, 45}:
            return position == 9
        elif mlp_0_0_output in {51}:
            return position == 63
        elif mlp_0_0_output in {56}:
            return position == 78
        elif mlp_0_0_output in {58}:
            return position == 11
        elif mlp_0_0_output in {60, 62}:
            return position == 32
        elif mlp_0_0_output in {61}:
            return position == 35
        elif mlp_0_0_output in {75}:
            return position == 4
        elif mlp_0_0_output in {78}:
            return position == 12
        elif mlp_0_0_output in {79}:
            return position == 14

    attn_1_1_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_7_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_1_output, attn_0_6_output):
        if mlp_0_1_output in {
            0,
            2,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            20,
            21,
            22,
            24,
            25,
            30,
            32,
            34,
            35,
            36,
            37,
            38,
            40,
            41,
            42,
            43,
            45,
            48,
            49,
            51,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            71,
            72,
            74,
            77,
            79,
        }:
            return attn_0_6_output == ")"
        elif mlp_0_1_output in {
            1,
            3,
            4,
            5,
            7,
            14,
            15,
            16,
            18,
            19,
            23,
            26,
            27,
            28,
            29,
            31,
            33,
            39,
            44,
            46,
            47,
            53,
            70,
            73,
            75,
            76,
            78,
        }:
            return attn_0_6_output == ""
        elif mlp_0_1_output in {17, 52}:
            return attn_0_6_output == "<s>"
        elif mlp_0_1_output in {50}:
            return attn_0_6_output == "("

    attn_1_2_pattern = select_closest(attn_0_6_outputs, mlp_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_7_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 62, 67, 38}:
            return position == 33
        elif mlp_0_0_output in {1}:
            return position == 43
        elif mlp_0_0_output in {2, 10}:
            return position == 44
        elif mlp_0_0_output in {3}:
            return position == 78
        elif mlp_0_0_output in {9, 4}:
            return position == 32
        elif mlp_0_0_output in {11, 5}:
            return position == 31
        elif mlp_0_0_output in {6}:
            return position == 22
        elif mlp_0_0_output in {7}:
            return position == 12
        elif mlp_0_0_output in {8, 46}:
            return position == 28
        elif mlp_0_0_output in {19, 12}:
            return position == 61
        elif mlp_0_0_output in {13}:
            return position == 76
        elif mlp_0_0_output in {14}:
            return position == 50
        elif mlp_0_0_output in {15}:
            return position == 40
        elif mlp_0_0_output in {
            32,
            66,
            37,
            39,
            41,
            73,
            74,
            60,
            16,
            17,
            54,
            24,
            57,
            26,
            28,
            29,
        }:
            return position == 3
        elif mlp_0_0_output in {59, 18, 51}:
            return position == 37
        elif mlp_0_0_output in {
            64,
            34,
            35,
            68,
            69,
            72,
            44,
            76,
            47,
            50,
            20,
            21,
            22,
            23,
            53,
            25,
            31,
        }:
            return position == 5
        elif mlp_0_0_output in {36, 70, 77, 79, 48, 52, 55, 27}:
            return position == 7
        elif mlp_0_0_output in {58, 30}:
            return position == 8
        elif mlp_0_0_output in {33}:
            return position == 79
        elif mlp_0_0_output in {40}:
            return position == 21
        elif mlp_0_0_output in {65, 71, 42, 78, 63}:
            return position == 17
        elif mlp_0_0_output in {43}:
            return position == 35
        elif mlp_0_0_output in {45}:
            return position == 9
        elif mlp_0_0_output in {49, 75}:
            return position == 4
        elif mlp_0_0_output in {56}:
            return position == 60
        elif mlp_0_0_output in {61}:
            return position == 55

    attn_1_3_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_6_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 12
        elif token in {"<s>"}:
            return position == 5

    attn_1_4_pattern = select_closest(positions, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_1_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 7}:
            return position == 22
        elif mlp_0_0_output in {1}:
            return position == 71
        elif mlp_0_0_output in {2}:
            return position == 75
        elif mlp_0_0_output in {3}:
            return position == 48
        elif mlp_0_0_output in {4}:
            return position == 33
        elif mlp_0_0_output in {8, 5, 62}:
            return position == 32
        elif mlp_0_0_output in {40, 6}:
            return position == 21
        elif mlp_0_0_output in {9}:
            return position == 26
        elif mlp_0_0_output in {10, 43, 14, 15}:
            return position == 37
        elif mlp_0_0_output in {11, 47}:
            return position == 28
        elif mlp_0_0_output in {12}:
            return position == 55
        elif mlp_0_0_output in {19, 13}:
            return position == 52
        elif mlp_0_0_output in {16, 18, 59, 22}:
            return position == 35
        elif mlp_0_0_output in {17}:
            return position == 2
        elif mlp_0_0_output in {20, 76}:
            return position == 29
        elif mlp_0_0_output in {32, 34, 67, 36, 37, 38, 51, 21, 25, 26, 27, 28}:
            return position == 5
        elif mlp_0_0_output in {68, 71, 46, 23}:
            return position == 31
        elif mlp_0_0_output in {24}:
            return position == 30
        elif mlp_0_0_output in {64, 29}:
            return position == 36
        elif mlp_0_0_output in {30}:
            return position == 6
        elif mlp_0_0_output in {57, 31}:
            return position == 8
        elif mlp_0_0_output in {33}:
            return position == 65
        elif mlp_0_0_output in {35, 55}:
            return position == 7
        elif mlp_0_0_output in {56, 39}:
            return position == 3
        elif mlp_0_0_output in {48, 41, 63}:
            return position == 9
        elif mlp_0_0_output in {42, 75, 78}:
            return position == 12
        elif mlp_0_0_output in {66, 44, 53}:
            return position == 24
        elif mlp_0_0_output in {52, 45}:
            return position == 19
        elif mlp_0_0_output in {49}:
            return position == 13
        elif mlp_0_0_output in {50}:
            return position == 11
        elif mlp_0_0_output in {77, 54}:
            return position == 18
        elif mlp_0_0_output in {58}:
            return position == 10
        elif mlp_0_0_output in {60}:
            return position == 14
        elif mlp_0_0_output in {61}:
            return position == 41
        elif mlp_0_0_output in {65}:
            return position == 17
        elif mlp_0_0_output in {72, 69}:
            return position == 27
        elif mlp_0_0_output in {70}:
            return position == 25
        elif mlp_0_0_output in {73, 79}:
            return position == 15
        elif mlp_0_0_output in {74}:
            return position == 45

    attn_1_5_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_7_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(mlp_0_0_output, attn_0_0_output):
        if mlp_0_0_output in {
            0,
            1,
            3,
            4,
            6,
            7,
            8,
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
            25,
            28,
            29,
            34,
            35,
            37,
            38,
            39,
            40,
            43,
            47,
            50,
            51,
            58,
            59,
            61,
            66,
            70,
            71,
        }:
            return attn_0_0_output == ""
        elif mlp_0_0_output in {
            2,
            5,
            9,
            10,
            21,
            23,
            24,
            26,
            27,
            30,
            31,
            32,
            33,
            36,
            41,
            42,
            44,
            45,
            46,
            48,
            49,
            52,
            53,
            54,
            55,
            56,
            57,
            60,
            62,
            63,
            64,
            65,
            67,
            68,
            69,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return attn_0_0_output == ")"
        elif mlp_0_0_output in {17}:
            return attn_0_0_output == "<s>"

    attn_1_6_pattern = select_closest(attn_0_0_outputs, mlp_0_0_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_6_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_2_output, position):
        if attn_0_2_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_2_output in {")"}:
            return position == 7

    attn_1_7_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_2_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_1_output, mlp_0_1_output):
        if num_mlp_0_1_output in {
            0,
            1,
            5,
            6,
            9,
            11,
            12,
            13,
            14,
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
            29,
            30,
            33,
            35,
            37,
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
            53,
            54,
            55,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            66,
            67,
            68,
            69,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return mlp_0_1_output == 2
        elif num_mlp_0_1_output in {
            32,
            65,
            2,
            3,
            34,
            36,
            38,
            7,
            8,
            10,
            15,
            16,
            18,
            52,
            56,
        }:
            return mlp_0_1_output == 41
        elif num_mlp_0_1_output in {4, 28, 70}:
            return mlp_0_1_output == 74
        elif num_mlp_0_1_output in {31}:
            return mlp_0_1_output == 12

    num_attn_1_0_pattern = select(
        mlp_0_1_outputs, num_mlp_0_1_outputs, num_predicate_1_0
    )
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_4_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {
            0,
            3,
            5,
            40,
            9,
            10,
            11,
            44,
            46,
            22,
            54,
            25,
            27,
            28,
            29,
            63,
        }:
            return mlp_0_0_output == 65
        elif num_mlp_0_1_output in {1, 74}:
            return mlp_0_0_output == 61
        elif num_mlp_0_1_output in {2, 41, 15, 53, 24, 58}:
            return mlp_0_0_output == 19
        elif num_mlp_0_1_output in {4, 36, 7, 13, 61}:
            return mlp_0_0_output == 15
        elif num_mlp_0_1_output in {6}:
            return mlp_0_0_output == 25
        elif num_mlp_0_1_output in {32, 8, 75, 23, 56}:
            return mlp_0_0_output == 33
        elif num_mlp_0_1_output in {73, 12, 30}:
            return mlp_0_0_output == 13
        elif num_mlp_0_1_output in {33, 67, 68, 70, 14, 47, 17, 49, 51, 21, 59, 62}:
            return mlp_0_0_output == 74
        elif num_mlp_0_1_output in {66, 71, 45, 79, 16, 50, 19, 26}:
            return mlp_0_0_output == 41
        elif num_mlp_0_1_output in {72, 18, 37}:
            return mlp_0_0_output == 28
        elif num_mlp_0_1_output in {34, 69, 39, 76, 20, 52, 60}:
            return mlp_0_0_output == 12
        elif num_mlp_0_1_output in {31}:
            return mlp_0_0_output == 27
        elif num_mlp_0_1_output in {35}:
            return mlp_0_0_output == 38
        elif num_mlp_0_1_output in {48, 38, 55}:
            return mlp_0_0_output == 2
        elif num_mlp_0_1_output in {42, 77}:
            return mlp_0_0_output == 59
        elif num_mlp_0_1_output in {57, 43, 78}:
            return mlp_0_0_output == 51
        elif num_mlp_0_1_output in {64}:
            return mlp_0_0_output == 78
        elif num_mlp_0_1_output in {65}:
            return mlp_0_0_output == 37

    num_attn_1_1_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_0_output, attn_0_4_output):
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
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return attn_0_4_output == ""

    num_attn_1_2_pattern = select(attn_0_4_outputs, mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_6_output, position):
        if attn_0_6_output in {"("}:
            return position == 64
        elif attn_0_6_output in {")"}:
            return position == 24
        elif attn_0_6_output in {"<s>"}:
            return position == 76

    num_attn_1_3_pattern = select(positions, attn_0_6_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_5_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_4_output, token):
        if attn_0_4_output in {"(", "<s>", ")"}:
            return token == ""

    num_attn_1_4_pattern = select(tokens, attn_0_4_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_3_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_5_output, mlp_0_1_output):
        if attn_0_5_output in {"("}:
            return mlp_0_1_output == 51
        elif attn_0_5_output in {")"}:
            return mlp_0_1_output == 18
        elif attn_0_5_output in {"<s>"}:
            return mlp_0_1_output == 8

    num_attn_1_5_pattern = select(mlp_0_1_outputs, attn_0_5_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(mlp_0_1_output, token):
        if mlp_0_1_output in {
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
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return token == ""
        elif mlp_0_1_output in {60}:
            return token == "<pad>"

    num_attn_1_6_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_2_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_4_output, mlp_0_0_output):
        if attn_0_4_output in {"("}:
            return mlp_0_0_output == 41
        elif attn_0_4_output in {"<s>", ")"}:
            return mlp_0_0_output == 60

    num_attn_1_7_pattern = select(mlp_0_0_outputs, attn_0_4_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_3_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_2_output, attn_0_5_output):
        key = (attn_0_2_output, attn_0_5_output)
        if key in {("(", "("), ("(", ")"), ("(", "<s>"), (")", "("), ("<s>", "(")}:
            return 42
        return 4

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_5_output, num_mlp_0_1_output):
        key = (attn_1_5_output, num_mlp_0_1_output)
        if key in {
            ("<s>", 0),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 19),
            ("<s>", 20),
            ("<s>", 25),
            ("<s>", 35),
            ("<s>", 38),
            ("<s>", 43),
            ("<s>", 51),
            ("<s>", 57),
            ("<s>", 61),
            ("<s>", 71),
            ("<s>", 76),
        }:
            return 48
        elif key in {("(", 75), (")", 75), ("<s>", 75)}:
            return 15
        elif key in {("<s>", 1)}:
            return 76
        return 17

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_5_outputs, num_mlp_0_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output):
        key = num_attn_1_0_output
        if key in {0}:
            return 24
        return 29

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_0_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_1_1_output):
        key = (num_attn_1_0_output, num_attn_1_1_output)
        return 66

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_1_output, position):
        if attn_0_1_output in {"("}:
            return position == 3
        elif attn_0_1_output in {")"}:
            return position == 13
        elif attn_0_1_output in {"<s>"}:
            return position == 1

    attn_2_0_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_4_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"(", ")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 5

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_7_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 66, 41, 50, 23}:
            return position == 8
        elif mlp_0_1_output in {1, 2, 3, 12}:
            return position == 41
        elif mlp_0_1_output in {32, 65, 4, 36, 71, 8, 73, 76, 46, 19, 55, 26}:
            return position == 12
        elif mlp_0_1_output in {5, 79, 48, 21, 54, 58, 27, 31}:
            return position == 9
        elif mlp_0_1_output in {6, 11, 45, 57, 30}:
            return position == 10
        elif mlp_0_1_output in {34, 69, 7, 39, 9, 40, 74, 13, 49, 51, 25, 59}:
            return position == 13
        elif mlp_0_1_output in {56, 10, 42, 61}:
            return position == 17
        elif mlp_0_1_output in {52, 14, 60}:
            return position == 14
        elif mlp_0_1_output in {77, 15}:
            return position == 15
        elif mlp_0_1_output in {16, 72, 35, 70}:
            return position == 11
        elif mlp_0_1_output in {17, 18}:
            return position == 7
        elif mlp_0_1_output in {20, 29}:
            return position == 4
        elif mlp_0_1_output in {22}:
            return position == 21
        elif mlp_0_1_output in {24}:
            return position == 5
        elif mlp_0_1_output in {75, 28}:
            return position == 3
        elif mlp_0_1_output in {33, 44, 47}:
            return position == 18
        elif mlp_0_1_output in {37}:
            return position == 24
        elif mlp_0_1_output in {67, 38, 63}:
            return position == 16
        elif mlp_0_1_output in {43}:
            return position == 22
        elif mlp_0_1_output in {53}:
            return position == 26
        elif mlp_0_1_output in {62}:
            return position == 23
        elif mlp_0_1_output in {64}:
            return position == 44
        elif mlp_0_1_output in {68}:
            return position == 28
        elif mlp_0_1_output in {78}:
            return position == 6

    attn_2_2_pattern = select_closest(positions, mlp_0_1_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {
            0,
            2,
            5,
            6,
            8,
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
            25,
            29,
            30,
            32,
            35,
            40,
            41,
            42,
            44,
            45,
            48,
            49,
            50,
            51,
            52,
            54,
            57,
            60,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            70,
            72,
            73,
            74,
            76,
            77,
            78,
            79,
        }:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {1, 10, 11, 39}:
            return mlp_0_1_output == 41
        elif mlp_0_0_output in {3}:
            return mlp_0_1_output == 52
        elif mlp_0_0_output in {4}:
            return mlp_0_1_output == 18
        elif mlp_0_0_output in {34, 7, 75, 53, 58}:
            return mlp_0_1_output == 17
        elif mlp_0_0_output in {9}:
            return mlp_0_1_output == 0
        elif mlp_0_0_output in {26, 15}:
            return mlp_0_1_output == 49
        elif mlp_0_0_output in {24}:
            return mlp_0_1_output == 43
        elif mlp_0_0_output in {27}:
            return mlp_0_1_output == 55
        elif mlp_0_0_output in {28}:
            return mlp_0_1_output == 14
        elif mlp_0_0_output in {31}:
            return mlp_0_1_output == 45
        elif mlp_0_0_output in {33}:
            return mlp_0_1_output == 44
        elif mlp_0_0_output in {36}:
            return mlp_0_1_output == 68
        elif mlp_0_0_output in {37}:
            return mlp_0_1_output == 59
        elif mlp_0_0_output in {38}:
            return mlp_0_1_output == 76
        elif mlp_0_0_output in {43}:
            return mlp_0_1_output == 51
        elif mlp_0_0_output in {46}:
            return mlp_0_1_output == 63
        elif mlp_0_0_output in {47}:
            return mlp_0_1_output == 21
        elif mlp_0_0_output in {55}:
            return mlp_0_1_output == 8
        elif mlp_0_0_output in {56}:
            return mlp_0_1_output == 74
        elif mlp_0_0_output in {59}:
            return mlp_0_1_output == 19
        elif mlp_0_0_output in {61}:
            return mlp_0_1_output == 65
        elif mlp_0_0_output in {69}:
            return mlp_0_1_output == 48
        elif mlp_0_0_output in {71}:
            return mlp_0_1_output == 75

    attn_2_3_pattern = select_closest(mlp_0_1_outputs, mlp_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_7_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 70, 8, 20, 57}:
            return k_mlp_0_0_output == 11
        elif q_mlp_0_0_output in {1, 2, 3, 39, 12, 15, 17, 23}:
            return k_mlp_0_0_output == 41
        elif q_mlp_0_0_output in {4, 36, 47}:
            return k_mlp_0_0_output == 7
        elif q_mlp_0_0_output in {5, 31}:
            return k_mlp_0_0_output == 8
        elif q_mlp_0_0_output in {46, 33, 6}:
            return k_mlp_0_0_output == 22
        elif q_mlp_0_0_output in {7, 79, 50, 56, 26}:
            return k_mlp_0_0_output == 10
        elif q_mlp_0_0_output in {72, 9, 73, 74, 45, 51, 21}:
            return k_mlp_0_0_output == 9
        elif q_mlp_0_0_output in {10}:
            return k_mlp_0_0_output == 20
        elif q_mlp_0_0_output in {27, 40, 11, 53}:
            return k_mlp_0_0_output == 16
        elif q_mlp_0_0_output in {35, 44, 13, 76, 78, 48, 19, 54}:
            return k_mlp_0_0_output == 15
        elif q_mlp_0_0_output in {32, 65, 68, 14}:
            return k_mlp_0_0_output == 18
        elif q_mlp_0_0_output in {71, 77, 16, 52, 55, 58, 61}:
            return k_mlp_0_0_output == 12
        elif q_mlp_0_0_output in {18}:
            return k_mlp_0_0_output == 27
        elif q_mlp_0_0_output in {37, 41, 22, 59, 30}:
            return k_mlp_0_0_output == 14
        elif q_mlp_0_0_output in {24}:
            return k_mlp_0_0_output == 60
        elif q_mlp_0_0_output in {25, 75, 43, 69}:
            return k_mlp_0_0_output == 17
        elif q_mlp_0_0_output in {28, 38}:
            return k_mlp_0_0_output == 19
        elif q_mlp_0_0_output in {49, 29}:
            return k_mlp_0_0_output == 6
        elif q_mlp_0_0_output in {34}:
            return k_mlp_0_0_output == 33
        elif q_mlp_0_0_output in {42}:
            return k_mlp_0_0_output == 23
        elif q_mlp_0_0_output in {60}:
            return k_mlp_0_0_output == 37
        elif q_mlp_0_0_output in {62}:
            return k_mlp_0_0_output == 26
        elif q_mlp_0_0_output in {63}:
            return k_mlp_0_0_output == 35
        elif q_mlp_0_0_output in {64}:
            return k_mlp_0_0_output == 31
        elif q_mlp_0_0_output in {66}:
            return k_mlp_0_0_output == 3
        elif q_mlp_0_0_output in {67}:
            return k_mlp_0_0_output == 4

    attn_2_4_pattern = select_closest(mlp_0_0_outputs, mlp_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 3
        elif attn_0_2_output in {")"}:
            return position == 5
        elif attn_0_2_output in {"<s>"}:
            return position == 1

    attn_2_5_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_4_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(mlp_0_0_output, position):
        if mlp_0_0_output in {0}:
            return position == 38
        elif mlp_0_0_output in {1, 2}:
            return position == 41
        elif mlp_0_0_output in {3}:
            return position == 59
        elif mlp_0_0_output in {9, 34, 4}:
            return position == 37
        elif mlp_0_0_output in {32, 5, 14}:
            return position == 24
        elif mlp_0_0_output in {64, 6, 42, 78, 31}:
            return position == 9
        elif mlp_0_0_output in {21, 7}:
            return position == 6
        elif mlp_0_0_output in {8, 44, 49, 50, 20, 24, 25}:
            return position == 8
        elif mlp_0_0_output in {10, 29}:
            return position == 23
        elif mlp_0_0_output in {51, 11, 28, 22}:
            return position == 19
        elif mlp_0_0_output in {12}:
            return position == 61
        elif mlp_0_0_output in {56, 75, 45, 13}:
            return position == 17
        elif mlp_0_0_output in {19, 15}:
            return position == 22
        elif mlp_0_0_output in {66, 76, 16, 48, 57, 58, 30}:
            return position == 10
        elif mlp_0_0_output in {17}:
            return position == 3
        elif mlp_0_0_output in {37, 40, 47, 18, 55}:
            return position == 18
        elif mlp_0_0_output in {23}:
            return position == 50
        elif mlp_0_0_output in {26, 60, 74}:
            return position == 13
        elif mlp_0_0_output in {27, 36}:
            return position == 11
        elif mlp_0_0_output in {33}:
            return position == 60
        elif mlp_0_0_output in {61, 35, 52, 53}:
            return position == 15
        elif mlp_0_0_output in {73, 67, 38, 71}:
            return position == 12
        elif mlp_0_0_output in {39}:
            return position == 7
        elif mlp_0_0_output in {65, 68, 69, 70, 41, 46, 59}:
            return position == 14
        elif mlp_0_0_output in {43}:
            return position == 21
        elif mlp_0_0_output in {54}:
            return position == 5
        elif mlp_0_0_output in {79, 62, 63}:
            return position == 20
        elif mlp_0_0_output in {72}:
            return position == 16
        elif mlp_0_0_output in {77}:
            return position == 26

    attn_2_6_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_4_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_2_output, mlp_0_1_output):
        if attn_0_2_output in {"(", ")"}:
            return mlp_0_1_output == 2
        elif attn_0_2_output in {"<s>"}:
            return mlp_0_1_output == 53

    attn_2_7_pattern = select_closest(mlp_0_1_outputs, attn_0_2_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_2_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_1_3_output):
        if attn_1_2_output in {"(", ")"}:
            return attn_1_3_output == ""
        elif attn_1_2_output in {"<s>"}:
            return attn_1_3_output == ")"

    num_attn_2_0_pattern = select(attn_1_3_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_0_1_output, token):
        if num_mlp_0_1_output in {
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
            40,
            41,
            42,
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
            60,
            61,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return token == ""
        elif num_mlp_0_1_output in {70, 43, 20, 22, 62}:
            return token == "<pad>"

    num_attn_2_1_pattern = select(tokens, num_mlp_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_6_output, mlp_1_0_output):
        if attn_1_6_output in {"("}:
            return mlp_1_0_output == 34
        elif attn_1_6_output in {")"}:
            return mlp_1_0_output == 60
        elif attn_1_6_output in {"<s>"}:
            return mlp_1_0_output == 2

    num_attn_2_2_pattern = select(mlp_1_0_outputs, attn_1_6_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_4_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_6_output, position):
        if attn_1_6_output in {"("}:
            return position == 5
        elif attn_1_6_output in {")"}:
            return position == 38
        elif attn_1_6_output in {"<s>"}:
            return position == 69

    num_attn_2_3_pattern = select(positions, attn_1_6_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(mlp_1_1_output, attn_1_3_output):
        if mlp_1_1_output in {
            0,
            1,
            2,
            3,
            5,
            7,
            8,
            9,
            12,
            14,
            16,
            19,
            20,
            21,
            23,
            24,
            26,
            28,
            29,
            30,
            33,
            35,
            36,
            39,
            40,
            41,
            42,
            43,
            45,
            46,
            50,
            51,
            53,
            54,
            56,
            57,
            61,
            62,
            63,
            66,
            69,
            70,
            71,
            74,
            79,
        }:
            return attn_1_3_output == ""
        elif mlp_1_1_output in {
            4,
            6,
            10,
            11,
            13,
            15,
            17,
            18,
            22,
            25,
            27,
            31,
            32,
            34,
            37,
            38,
            44,
            47,
            48,
            49,
            52,
            55,
            58,
            59,
            60,
            64,
            65,
            67,
            68,
            73,
            75,
            76,
            77,
            78,
        }:
            return attn_1_3_output == ")"
        elif mlp_1_1_output in {72}:
            return attn_1_3_output == "<s>"

    num_attn_2_4_pattern = select(attn_1_3_outputs, mlp_1_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_7_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_5_output, attn_1_7_output):
        if attn_1_5_output in {"(", "<s>", ")"}:
            return attn_1_7_output == ""

    num_attn_2_5_pattern = select(attn_1_7_outputs, attn_1_5_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_2_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_6_output, attn_1_0_output):
        if attn_1_6_output in {"(", "<s>", ")"}:
            return attn_1_0_output == ""

    num_attn_2_6_pattern = select(attn_1_0_outputs, attn_1_6_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_1_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_2_output, attn_1_4_output):
        if attn_1_2_output in {"(", ")"}:
            return attn_1_4_output == ""
        elif attn_1_2_output in {"<s>"}:
            return attn_1_4_output == "<pad>"

    num_attn_2_7_pattern = select(attn_1_4_outputs, attn_1_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_5_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, attn_2_3_output):
        key = (attn_2_2_output, attn_2_3_output)
        return 0

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_5_output, attn_2_1_output):
        key = (attn_2_5_output, attn_2_1_output)
        if key in {("(", ")"), (")", "("), (")", ")"), (")", "<s>")}:
            return 64
        elif key in {("(", "("), ("(", "<s>"), ("<s>", "(")}:
            return 20
        elif key in {("<s>", ")"), ("<s>", "<s>")}:
            return 59
        return 56

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_5_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_7_output, num_attn_0_7_output):
        key = (num_attn_2_7_output, num_attn_0_7_output)
        return 44

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output, num_attn_1_3_output):
        key = (num_attn_2_1_output, num_attn_1_3_output)
        return 15

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_3_outputs)
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
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            ")",
            ")",
            "(",
            "(",
            ")",
            ")",
            ")",
            ")",
            "(",
            ")",
            "(",
            ")",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            "(",
        ]
    )
)
