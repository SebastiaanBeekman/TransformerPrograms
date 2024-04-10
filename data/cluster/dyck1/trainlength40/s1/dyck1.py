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
        "output/length/rasp/dyck1/trainlength40/s1/dyck1_weights.csv",
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
        if q_position in {0, 13}:
            return k_position == 37
        elif q_position in {1, 9}:
            return k_position == 61
        elif q_position in {2, 10}:
            return k_position == 8
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {6, 39}:
            return k_position == 4
        elif q_position in {76, 71, 7}:
            return k_position == 7
        elif q_position in {8, 27}:
            return k_position == 6
        elif q_position in {11, 44}:
            return k_position == 49
        elif q_position in {34, 12}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {19, 68, 77, 15}:
            return k_position == 11
        elif q_position in {33, 36, 38, 16, 18}:
            return k_position == 12
        elif q_position in {17, 21, 23}:
            return k_position == 14
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {66, 52, 22}:
            return k_position == 18
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25, 46}:
            return k_position == 17
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {43, 75, 53, 54, 57, 60, 30}:
            return k_position == 9
        elif q_position in {32, 31}:
            return k_position == 30
        elif q_position in {35}:
            return k_position == 31
        elif q_position in {56, 37}:
            return k_position == 26
        elif q_position in {40}:
            return k_position == 60
        elif q_position in {41}:
            return k_position == 52
        elif q_position in {42}:
            return k_position == 1
        elif q_position in {45}:
            return k_position == 44
        elif q_position in {47}:
            return k_position == 45
        elif q_position in {48}:
            return k_position == 40
        elif q_position in {49}:
            return k_position == 29
        elif q_position in {50}:
            return k_position == 41
        elif q_position in {51}:
            return k_position == 62
        elif q_position in {55}:
            return k_position == 66
        elif q_position in {58}:
            return k_position == 56
        elif q_position in {59, 63}:
            return k_position == 74
        elif q_position in {61}:
            return k_position == 64
        elif q_position in {62}:
            return k_position == 72
        elif q_position in {64}:
            return k_position == 46
        elif q_position in {65}:
            return k_position == 65
        elif q_position in {67}:
            return k_position == 21
        elif q_position in {69}:
            return k_position == 35
        elif q_position in {70}:
            return k_position == 71
        elif q_position in {72}:
            return k_position == 50
        elif q_position in {73}:
            return k_position == 47
        elif q_position in {74}:
            return k_position == 59
        elif q_position in {78}:
            return k_position == 38
        elif q_position in {79}:
            return k_position == 77

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 66, 43, 13, 51}:
            return k_position == 12
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 74
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5, 6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9, 41}:
            return k_position == 63
        elif q_position in {10, 39}:
            return k_position == 8
        elif q_position in {70, 58, 11, 46}:
            return k_position == 9
        elif q_position in {12, 14}:
            return k_position == 10
        elif q_position in {59, 15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17, 47}:
            return k_position == 16
        elif q_position in {65, 18, 54}:
            return k_position == 17
        elif q_position in {64, 19}:
            return k_position == 18
        elif q_position in {20, 76}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {68, 22}:
            return k_position == 21
        elif q_position in {25, 74, 23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {42, 29}:
            return k_position == 28
        elif q_position in {30}:
            return k_position == 29
        elif q_position in {45, 31}:
            return k_position == 30
        elif q_position in {32}:
            return k_position == 31
        elif q_position in {33, 49}:
            return k_position == 32
        elif q_position in {34, 36}:
            return k_position == 33
        elif q_position in {75, 35, 78}:
            return k_position == 34
        elif q_position in {44, 37}:
            return k_position == 36
        elif q_position in {53, 38}:
            return k_position == 35
        elif q_position in {40}:
            return k_position == 42
        elif q_position in {48}:
            return k_position == 45
        elif q_position in {50}:
            return k_position == 48
        elif q_position in {52}:
            return k_position == 68
        elif q_position in {73, 55}:
            return k_position == 11
        elif q_position in {56}:
            return k_position == 64
        elif q_position in {57}:
            return k_position == 59
        elif q_position in {60}:
            return k_position == 24
        elif q_position in {67, 61}:
            return k_position == 39
        elif q_position in {62, 63}:
            return k_position == 70
        elif q_position in {69}:
            return k_position == 38
        elif q_position in {71}:
            return k_position == 41
        elif q_position in {72}:
            return k_position == 46
        elif q_position in {77}:
            return k_position == 60
        elif q_position in {79}:
            return k_position == 78

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
            return position == 8
        elif token in {"<s>"}:
            return position == 2

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 3

    attn_0_3_pattern = select_closest(positions, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 72}:
            return k_position == 68
        elif q_position in {1, 43}:
            return k_position == 64
        elif q_position in {8, 2, 39}:
            return k_position == 6
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {57, 5}:
            return k_position == 37
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 69
        elif q_position in {9, 53}:
            return k_position == 23
        elif q_position in {10, 28}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 43
        elif q_position in {70, 12, 77, 14, 15}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 78
        elif q_position in {16}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 14
        elif q_position in {18}:
            return k_position == 15
        elif q_position in {64, 19, 44, 78}:
            return k_position == 7
        elif q_position in {46, 42, 20, 22}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 19
        elif q_position in {58, 23}:
            return k_position == 20
        elif q_position in {24, 63}:
            return k_position == 22
        elif q_position in {25, 69}:
            return k_position == 16
        elif q_position in {73, 26, 29}:
            return k_position == 25
        elif q_position in {27, 47}:
            return k_position == 17
        elif q_position in {65, 68, 75, 49, 51, 30}:
            return k_position == 9
        elif q_position in {31}:
            return k_position == 24
        elif q_position in {32, 34}:
            return k_position == 10
        elif q_position in {33, 74}:
            return k_position == 29
        elif q_position in {35}:
            return k_position == 31
        elif q_position in {36, 37}:
            return k_position == 34
        elif q_position in {38}:
            return k_position == 36
        elif q_position in {40}:
            return k_position == 61
        elif q_position in {41, 67}:
            return k_position == 27
        elif q_position in {45}:
            return k_position == 62
        elif q_position in {48, 79}:
            return k_position == 79
        elif q_position in {50}:
            return k_position == 67
        elif q_position in {52}:
            return k_position == 46
        elif q_position in {54}:
            return k_position == 28
        elif q_position in {55}:
            return k_position == 1
        elif q_position in {56}:
            return k_position == 47
        elif q_position in {59}:
            return k_position == 41
        elif q_position in {60}:
            return k_position == 51
        elif q_position in {61}:
            return k_position == 52
        elif q_position in {62}:
            return k_position == 32
        elif q_position in {66}:
            return k_position == 66
        elif q_position in {71}:
            return k_position == 74
        elif q_position in {76}:
            return k_position == 58

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 17, 13, 47}:
            return k_position == 57
        elif q_position in {1, 60}:
            return k_position == 69
        elif q_position in {8, 2}:
            return k_position == 6
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 45
        elif q_position in {43, 6}:
            return k_position == 5
        elif q_position in {35, 44, 7}:
            return k_position == 33
        elif q_position in {9, 69}:
            return k_position == 47
        elif q_position in {41, 10, 26}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 75
        elif q_position in {32, 12, 36, 15}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {16, 19, 76}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 16
        elif q_position in {20, 21, 23}:
            return k_position == 18
        elif q_position in {22}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 22
        elif q_position in {27, 30}:
            return k_position == 21
        elif q_position in {28}:
            return k_position == 8
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {52, 70, 31}:
            return k_position == 25
        elif q_position in {33}:
            return k_position == 27
        elif q_position in {34, 38, 73, 55, 63}:
            return k_position == 11
        elif q_position in {56, 37, 71}:
            return k_position == 35
        elif q_position in {39, 40, 48, 53, 54, 58}:
            return k_position == 7
        elif q_position in {64, 42, 68}:
            return k_position == 1
        elif q_position in {45}:
            return k_position == 19
        elif q_position in {78, 46}:
            return k_position == 39
        elif q_position in {49, 57}:
            return k_position == 65
        elif q_position in {50}:
            return k_position == 73
        elif q_position in {51}:
            return k_position == 67
        elif q_position in {59, 77, 79}:
            return k_position == 70
        elif q_position in {61}:
            return k_position == 56
        elif q_position in {62}:
            return k_position == 58
        elif q_position in {65}:
            return k_position == 30
        elif q_position in {66}:
            return k_position == 68
        elif q_position in {67}:
            return k_position == 72
        elif q_position in {72}:
            return k_position == 59
        elif q_position in {74}:
            return k_position == 74
        elif q_position in {75}:
            return k_position == 38

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 65, 31}:
            return k_position == 30
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 48
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 5
        elif q_position in {8, 7}:
            return k_position == 7
        elif q_position in {9, 10, 75, 78, 51, 53}:
            return k_position == 9
        elif q_position in {73, 57, 34, 11}:
            return k_position == 10
        elif q_position in {32, 36, 70, 43, 12, 13, 46, 55, 59}:
            return k_position == 11
        elif q_position in {33, 14}:
            return k_position == 13
        elif q_position in {72, 28, 30, 15}:
            return k_position == 12
        elif q_position in {16, 19, 23}:
            return k_position == 14
        elif q_position in {48, 17, 42, 76}:
            return k_position == 16
        elif q_position in {41, 47, 79, 18, 56, 29, 63}:
            return k_position == 15
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 18
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {24, 49}:
            return k_position == 23
        elif q_position in {77, 45, 50, 25, 26}:
            return k_position == 22
        elif q_position in {67, 27, 52}:
            return k_position == 24
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 6
        elif q_position in {38}:
            return k_position == 37
        elif q_position in {39}:
            return k_position == 8
        elif q_position in {40}:
            return k_position == 27
        elif q_position in {44}:
            return k_position == 46
        elif q_position in {54}:
            return k_position == 41
        elif q_position in {64, 58, 74, 71}:
            return k_position == 31
        elif q_position in {60}:
            return k_position == 47
        elif q_position in {61}:
            return k_position == 56
        elif q_position in {62}:
            return k_position == 61
        elif q_position in {66}:
            return k_position == 77
        elif q_position in {68}:
            return k_position == 32
        elif q_position in {69}:
            return k_position == 75

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 44
        elif q_position in {1}:
            return k_position == 63
        elif q_position in {32, 2, 14}:
            return k_position == 12
        elif q_position in {56, 3, 4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 57
        elif q_position in {6, 7}:
            return k_position == 4
        elif q_position in {8, 22}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 70
        elif q_position in {64, 10, 58, 47}:
            return k_position == 7
        elif q_position in {11, 12, 69}:
            return k_position == 9
        elif q_position in {34, 13, 23}:
            return k_position == 11
        elif q_position in {17, 15}:
            return k_position == 13
        elif q_position in {16, 41, 74, 19}:
            return k_position == 15
        elif q_position in {18, 20, 21}:
            return k_position == 17
        elif q_position in {43, 78, 51, 24, 30}:
            return k_position == 21
        elif q_position in {25, 59, 45}:
            return k_position == 19
        elif q_position in {26, 27, 42, 68}:
            return k_position == 25
        elif q_position in {28}:
            return k_position == 26
        elif q_position in {40, 61, 29, 46}:
            return k_position == 22
        elif q_position in {31}:
            return k_position == 30
        elif q_position in {33}:
            return k_position == 29
        elif q_position in {35, 52, 79}:
            return k_position == 32
        elif q_position in {36}:
            return k_position == 35
        elif q_position in {37, 39}:
            return k_position == 33
        elif q_position in {38}:
            return k_position == 36
        elif q_position in {44}:
            return k_position == 74
        elif q_position in {48}:
            return k_position == 73
        elif q_position in {49, 53}:
            return k_position == 58
        elif q_position in {50}:
            return k_position == 3
        elif q_position in {54}:
            return k_position == 54
        elif q_position in {55}:
            return k_position == 39
        elif q_position in {57, 60}:
            return k_position == 78
        elif q_position in {62}:
            return k_position == 60
        elif q_position in {63}:
            return k_position == 49
        elif q_position in {65, 77}:
            return k_position == 23
        elif q_position in {72, 66}:
            return k_position == 38
        elif q_position in {73, 67}:
            return k_position == 61
        elif q_position in {70}:
            return k_position == 27
        elif q_position in {71}:
            return k_position == 46
        elif q_position in {75}:
            return k_position == 43
        elif q_position in {76}:
            return k_position == 24

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 65, 67}:
            return k_position == 75
        elif q_position in {1, 35, 5, 47}:
            return k_position == 62
        elif q_position in {2}:
            return k_position == 60
        elif q_position in {3, 31}:
            return k_position == 64
        elif q_position in {56, 4}:
            return k_position == 58
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {18, 7}:
            return k_position == 63
        elif q_position in {8}:
            return k_position == 52
        elif q_position in {9, 63}:
            return k_position == 61
        elif q_position in {64, 10, 79}:
            return k_position == 2
        elif q_position in {11}:
            return k_position == 68
        elif q_position in {12, 23}:
            return k_position == 29
        elif q_position in {13}:
            return k_position == 25
        elif q_position in {46, 45, 14}:
            return k_position == 79
        elif q_position in {24, 15}:
            return k_position == 53
        elif q_position in {66, 77, 16, 50, 59}:
            return k_position == 71
        elif q_position in {17, 22}:
            return k_position == 31
        elif q_position in {19}:
            return k_position == 35
        elif q_position in {20}:
            return k_position == 22
        elif q_position in {39, 42, 21, 57, 61}:
            return k_position == 77
        elif q_position in {25}:
            return k_position == 40
        elif q_position in {26, 38}:
            return k_position == 59
        elif q_position in {72, 27}:
            return k_position == 49
        elif q_position in {28, 53}:
            return k_position == 78
        elif q_position in {29}:
            return k_position == 48
        elif q_position in {30}:
            return k_position == 51
        elif q_position in {32}:
            return k_position == 33
        elif q_position in {33, 68}:
            return k_position == 70
        elif q_position in {48, 34}:
            return k_position == 42
        elif q_position in {49, 36}:
            return k_position == 55
        elif q_position in {37}:
            return k_position == 39
        elif q_position in {40}:
            return k_position == 72
        elif q_position in {41, 76}:
            return k_position == 10
        elif q_position in {43, 78}:
            return k_position == 6
        elif q_position in {44}:
            return k_position == 8
        elif q_position in {51}:
            return k_position == 45
        elif q_position in {74, 75, 52, 70}:
            return k_position == 13
        elif q_position in {54, 55}:
            return k_position == 43
        elif q_position in {58}:
            return k_position == 15
        elif q_position in {60}:
            return k_position == 57
        elif q_position in {62}:
            return k_position == 41
        elif q_position in {69}:
            return k_position == 54
        elif q_position in {71}:
            return k_position == 3
        elif q_position in {73}:
            return k_position == 66

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 16}:
            return k_position == 36
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2, 31}:
            return k_position == 39
        elif q_position in {17, 3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 51
        elif q_position in {41, 20, 5}:
            return k_position == 70
        elif q_position in {54, 78, 75, 6}:
            return k_position == 73
        elif q_position in {44, 7}:
            return k_position == 59
        elif q_position in {8, 64}:
            return k_position == 10
        elif q_position in {9, 19}:
            return k_position == 25
        elif q_position in {10}:
            return k_position == 18
        elif q_position in {11}:
            return k_position == 56
        elif q_position in {12, 45}:
            return k_position == 65
        elif q_position in {13}:
            return k_position == 72
        elif q_position in {51, 14}:
            return k_position == 62
        elif q_position in {15}:
            return k_position == 26
        elif q_position in {18}:
            return k_position == 22
        elif q_position in {21}:
            return k_position == 75
        elif q_position in {70, 61, 22}:
            return k_position == 52
        elif q_position in {37, 30, 23}:
            return k_position == 77
        elif q_position in {24}:
            return k_position == 64
        elif q_position in {25}:
            return k_position == 21
        elif q_position in {26, 27}:
            return k_position == 53
        elif q_position in {28}:
            return k_position == 54
        elif q_position in {77, 29}:
            return k_position == 46
        elif q_position in {32}:
            return k_position == 61
        elif q_position in {33}:
            return k_position == 33
        elif q_position in {65, 34}:
            return k_position == 79
        elif q_position in {35}:
            return k_position == 24
        elif q_position in {36}:
            return k_position == 55
        elif q_position in {56, 38, 47}:
            return k_position == 66
        elif q_position in {39}:
            return k_position == 43
        elif q_position in {40}:
            return k_position == 68
        elif q_position in {49, 42, 62}:
            return k_position == 69
        elif q_position in {73, 50, 43}:
            return k_position == 41
        elif q_position in {72, 46}:
            return k_position == 42
        elif q_position in {48}:
            return k_position == 9
        elif q_position in {52}:
            return k_position == 58
        elif q_position in {67, 53}:
            return k_position == 60
        elif q_position in {66, 55}:
            return k_position == 49
        elif q_position in {57, 74}:
            return k_position == 50
        elif q_position in {58}:
            return k_position == 74
        elif q_position in {59}:
            return k_position == 45
        elif q_position in {60}:
            return k_position == 48
        elif q_position in {63}:
            return k_position == 63
        elif q_position in {68}:
            return k_position == 15
        elif q_position in {69}:
            return k_position == 71
        elif q_position in {71}:
            return k_position == 67
        elif q_position in {76}:
            return k_position == 76
        elif q_position in {79}:
            return k_position == 4

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 40
        elif q_position in {1}:
            return k_position == 31
        elif q_position in {2, 67, 71}:
            return k_position == 63
        elif q_position in {3}:
            return k_position == 33
        elif q_position in {34, 4}:
            return k_position == 48
        elif q_position in {5}:
            return k_position == 52
        elif q_position in {74, 6}:
            return k_position == 53
        elif q_position in {7}:
            return k_position == 34
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 28
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12, 31}:
            return k_position == 46
        elif q_position in {60, 13, 30}:
            return k_position == 62
        elif q_position in {14}:
            return k_position == 55
        elif q_position in {50, 15}:
            return k_position == 37
        elif q_position in {16}:
            return k_position == 21
        elif q_position in {17, 25}:
            return k_position == 26
        elif q_position in {18}:
            return k_position == 66
        elif q_position in {19}:
            return k_position == 36
        elif q_position in {49, 20, 53, 36}:
            return k_position == 47
        elif q_position in {29, 21}:
            return k_position == 61
        elif q_position in {22}:
            return k_position == 29
        elif q_position in {23}:
            return k_position == 59
        elif q_position in {24}:
            return k_position == 58
        elif q_position in {26, 55}:
            return k_position == 72
        elif q_position in {27}:
            return k_position == 35
        elif q_position in {65, 28}:
            return k_position == 70
        elif q_position in {32, 64, 63}:
            return k_position == 51
        elif q_position in {33, 52, 41}:
            return k_position == 64
        elif q_position in {35, 69, 78, 39}:
            return k_position == 65
        elif q_position in {68, 37}:
            return k_position == 42
        elif q_position in {38}:
            return k_position == 78
        elif q_position in {40, 48}:
            return k_position == 49
        elif q_position in {42, 54}:
            return k_position == 57
        elif q_position in {43}:
            return k_position == 73
        elif q_position in {44, 70}:
            return k_position == 50
        elif q_position in {45}:
            return k_position == 68
        elif q_position in {46}:
            return k_position == 79
        elif q_position in {47}:
            return k_position == 77
        elif q_position in {51, 76}:
            return k_position == 41
        elif q_position in {56}:
            return k_position == 39
        elif q_position in {57}:
            return k_position == 43
        elif q_position in {58}:
            return k_position == 75
        elif q_position in {75, 59, 79}:
            return k_position == 44
        elif q_position in {61}:
            return k_position == 71
        elif q_position in {62}:
            return k_position == 74
        elif q_position in {66, 77}:
            return k_position == 69
        elif q_position in {72}:
            return k_position == 1
        elif q_position in {73}:
            return k_position == 20

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
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
        elif position in {1, 3, 5, 7, 9, 11}:
            return token == ")"
        elif position in {13}:
            return token == "<pad>"
        elif position in {37, 29}:
            return token == "<s>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0, 17}:
            return k_position == 22
        elif q_position in {1, 57}:
            return k_position == 46
        elif q_position in {2}:
            return k_position == 76
        elif q_position in {3, 6}:
            return k_position == 27
        elif q_position in {49, 4, 30}:
            return k_position == 52
        elif q_position in {5}:
            return k_position == 38
        elif q_position in {7}:
            return k_position == 69
        elif q_position in {8, 16}:
            return k_position == 23
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10, 66}:
            return k_position == 78
        elif q_position in {11}:
            return k_position == 37
        elif q_position in {51, 19, 12, 21}:
            return k_position == 77
        elif q_position in {73, 29, 67, 13}:
            return k_position == 40
        elif q_position in {14}:
            return k_position == 60
        elif q_position in {26, 15}:
            return k_position == 36
        elif q_position in {40, 18, 47}:
            return k_position == 48
        elif q_position in {20}:
            return k_position == 0
        elif q_position in {72, 22}:
            return k_position == 50
        elif q_position in {34, 36, 23}:
            return k_position == 44
        elif q_position in {24}:
            return k_position == 71
        elif q_position in {25}:
            return k_position == 34
        elif q_position in {27, 53}:
            return k_position == 51
        elif q_position in {28}:
            return k_position == 33
        elif q_position in {31}:
            return k_position == 39
        elif q_position in {32, 77}:
            return k_position == 73
        elif q_position in {33, 63}:
            return k_position == 24
        elif q_position in {35}:
            return k_position == 45
        elif q_position in {37}:
            return k_position == 64
        elif q_position in {38}:
            return k_position == 41
        elif q_position in {39}:
            return k_position == 65
        elif q_position in {65, 70, 41, 76, 78}:
            return k_position == 13
        elif q_position in {42}:
            return k_position == 72
        elif q_position in {43}:
            return k_position == 12
        elif q_position in {56, 44, 45}:
            return k_position == 8
        elif q_position in {62, 46}:
            return k_position == 63
        elif q_position in {48}:
            return k_position == 14
        elif q_position in {64, 50}:
            return k_position == 70
        elif q_position in {52}:
            return k_position == 21
        elif q_position in {59, 54}:
            return k_position == 62
        elif q_position in {55}:
            return k_position == 2
        elif q_position in {58}:
            return k_position == 47
        elif q_position in {60, 68}:
            return k_position == 16
        elif q_position in {61}:
            return k_position == 17
        elif q_position in {69}:
            return k_position == 18
        elif q_position in {71}:
            return k_position == 11
        elif q_position in {74}:
            return k_position == 7
        elif q_position in {75}:
            return k_position == 59
        elif q_position in {79}:
            return k_position == 15

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {
            0,
            7,
            9,
            11,
            15,
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
            return token == ""
        elif position in {1, 3, 4, 5, 6}:
            return token == ")"
        elif position in {2, 8, 10, 12, 13, 14, 16}:
            return token == "<s>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"("}:
            return position == 57
        elif token in {")"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 54

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_token, k_token):
        if q_token in {"("}:
            return k_token == "<s>"
        elif q_token in {")"}:
            return k_token == ""
        elif q_token in {"<s>"}:
            return k_token == ")"

    num_attn_0_7_pattern = select(tokens, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_4_output):
        key = (attn_0_0_output, attn_0_4_output)
        if key in {("<s>", "("), ("<s>", "<s>")}:
            return 72
        return 13

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_4_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_6_output):
        key = (attn_0_1_output, attn_0_6_output)
        return 55

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_6_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_6_output):
        key = (num_attn_0_1_output, num_attn_0_6_output)
        return 0

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_5_output, num_attn_0_0_output):
        key = (num_attn_0_5_output, num_attn_0_0_output)
        return 63

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 1, 34, 7, 14, 47, 48}:
            return position == 35
        elif mlp_0_0_output in {2}:
            return position == 48
        elif mlp_0_0_output in {
            3,
            5,
            8,
            12,
            18,
            20,
            25,
            39,
            41,
            50,
            51,
            53,
            55,
            57,
            58,
            59,
            67,
            68,
            70,
            72,
            74,
            75,
            77,
            78,
        }:
            return position == 1
        elif mlp_0_0_output in {4}:
            return position == 61
        elif mlp_0_0_output in {6}:
            return position == 57
        elif mlp_0_0_output in {66, 9, 19, 56, 61}:
            return position == 31
        elif mlp_0_0_output in {64, 35, 40, 73, 10, 52, 26, 27, 28, 30, 63}:
            return position == 33
        elif mlp_0_0_output in {11, 15}:
            return position == 41
        elif mlp_0_0_output in {16, 29, 13}:
            return position == 3
        elif mlp_0_0_output in {17, 42, 31}:
            return position == 37
        elif mlp_0_0_output in {21}:
            return position == 50
        elif mlp_0_0_output in {22}:
            return position == 69
        elif mlp_0_0_output in {23}:
            return position == 26
        elif mlp_0_0_output in {24}:
            return position == 20
        elif mlp_0_0_output in {32}:
            return position == 25
        elif mlp_0_0_output in {33}:
            return position == 34
        elif mlp_0_0_output in {36}:
            return position == 43
        elif mlp_0_0_output in {54, 37, 46}:
            return position == 32
        elif mlp_0_0_output in {44, 38}:
            return position == 29
        elif mlp_0_0_output in {43}:
            return position == 53
        elif mlp_0_0_output in {76, 45}:
            return position == 39
        elif mlp_0_0_output in {65, 49}:
            return position == 36
        elif mlp_0_0_output in {60}:
            return position == 27
        elif mlp_0_0_output in {62}:
            return position == 4
        elif mlp_0_0_output in {69}:
            return position == 30
        elif mlp_0_0_output in {71}:
            return position == 8
        elif mlp_0_0_output in {79}:
            return position == 56

    attn_1_0_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"<s>", "("}:
            return position == 1
        elif token in {")"}:
            return position == 6

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_7_output, position):
        if attn_0_7_output in {"<s>", "("}:
            return position == 1
        elif attn_0_7_output in {")"}:
            return position == 3

    attn_1_2_pattern = select_closest(positions, attn_0_7_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 1
        elif attn_0_2_output in {")"}:
            return position == 29
        elif attn_0_2_output in {"<s>"}:
            return position == 3

    attn_1_3_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 1, 3, 69, 10, 49, 25}:
            return position == 1
        elif mlp_0_1_output in {2}:
            return position == 33
        elif mlp_0_1_output in {4}:
            return position == 49
        elif mlp_0_1_output in {5}:
            return position == 26
        elif mlp_0_1_output in {6}:
            return position == 30
        elif mlp_0_1_output in {9, 19, 76, 7}:
            return position == 34
        elif mlp_0_1_output in {8}:
            return position == 70
        elif mlp_0_1_output in {11}:
            return position == 53
        elif mlp_0_1_output in {12}:
            return position == 44
        elif mlp_0_1_output in {13}:
            return position == 58
        elif mlp_0_1_output in {14}:
            return position == 68
        elif mlp_0_1_output in {15}:
            return position == 36
        elif mlp_0_1_output in {16}:
            return position == 27
        elif mlp_0_1_output in {17}:
            return position == 47
        elif mlp_0_1_output in {18}:
            return position == 29
        elif mlp_0_1_output in {67, 20, 78}:
            return position == 3
        elif mlp_0_1_output in {21, 31}:
            return position == 71
        elif mlp_0_1_output in {70, 71, 41, 43, 47, 52, 22}:
            return position == 5
        elif mlp_0_1_output in {23}:
            return position == 76
        elif mlp_0_1_output in {24}:
            return position == 59
        elif mlp_0_1_output in {26}:
            return position == 13
        elif mlp_0_1_output in {27}:
            return position == 65
        elif mlp_0_1_output in {28, 37}:
            return position == 10
        elif mlp_0_1_output in {65, 61, 29}:
            return position == 17
        elif mlp_0_1_output in {30}:
            return position == 57
        elif mlp_0_1_output in {32}:
            return position == 28
        elif mlp_0_1_output in {33}:
            return position == 11
        elif mlp_0_1_output in {73, 34}:
            return position == 37
        elif mlp_0_1_output in {58, 35}:
            return position == 50
        elif mlp_0_1_output in {36, 79}:
            return position == 25
        elif mlp_0_1_output in {38}:
            return position == 74
        elif mlp_0_1_output in {66, 51, 60, 39}:
            return position == 7
        elif mlp_0_1_output in {40}:
            return position == 2
        elif mlp_0_1_output in {42}:
            return position == 60
        elif mlp_0_1_output in {48, 64, 44}:
            return position == 19
        elif mlp_0_1_output in {45}:
            return position == 9
        elif mlp_0_1_output in {57, 46}:
            return position == 35
        elif mlp_0_1_output in {50, 75, 63}:
            return position == 75
        elif mlp_0_1_output in {53}:
            return position == 32
        elif mlp_0_1_output in {54}:
            return position == 22
        elif mlp_0_1_output in {55}:
            return position == 8
        elif mlp_0_1_output in {56}:
            return position == 61
        elif mlp_0_1_output in {59}:
            return position == 69
        elif mlp_0_1_output in {62}:
            return position == 66
        elif mlp_0_1_output in {68}:
            return position == 78
        elif mlp_0_1_output in {72}:
            return position == 51
        elif mlp_0_1_output in {74}:
            return position == 42
        elif mlp_0_1_output in {77}:
            return position == 41

    attn_1_4_pattern = select_closest(positions, mlp_0_1_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_7_output, position):
        if attn_0_7_output in {"("}:
            return position == 60
        elif attn_0_7_output in {")"}:
            return position == 3
        elif attn_0_7_output in {"<s>"}:
            return position == 64

    attn_1_5_pattern = select_closest(positions, attn_0_7_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_6_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(token, position):
        if token in {"("}:
            return position == 7
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 1

    attn_1_6_pattern = select_closest(positions, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_3_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 7
        elif attn_0_4_output in {")"}:
            return position == 78
        elif attn_0_4_output in {"<s>"}:
            return position == 1

    attn_1_7_pattern = select_closest(positions, attn_0_4_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_0_output, mlp_0_1_output):
        if mlp_0_0_output in {0, 66, 5, 55}:
            return mlp_0_1_output == 49
        elif mlp_0_0_output in {1, 51}:
            return mlp_0_1_output == 65
        elif mlp_0_0_output in {2}:
            return mlp_0_1_output == 20
        elif mlp_0_0_output in {35, 3, 7}:
            return mlp_0_1_output == 22
        elif mlp_0_0_output in {17, 4, 6, 14}:
            return mlp_0_1_output == 35
        elif mlp_0_0_output in {8}:
            return mlp_0_1_output == 67
        elif mlp_0_0_output in {9}:
            return mlp_0_1_output == 6
        elif mlp_0_0_output in {57, 10, 38, 71}:
            return mlp_0_1_output == 27
        elif mlp_0_0_output in {75, 73, 67, 11}:
            return mlp_0_1_output == 59
        elif mlp_0_0_output in {12}:
            return mlp_0_1_output == 41
        elif mlp_0_0_output in {65, 29, 13, 62}:
            return mlp_0_1_output == 1
        elif mlp_0_0_output in {61, 15}:
            return mlp_0_1_output == 68
        elif mlp_0_0_output in {16, 64}:
            return mlp_0_1_output == 46
        elif mlp_0_0_output in {18}:
            return mlp_0_1_output == 42
        elif mlp_0_0_output in {19}:
            return mlp_0_1_output == 63
        elif mlp_0_0_output in {20, 44, 23}:
            return mlp_0_1_output == 30
        elif mlp_0_0_output in {74, 21, 78}:
            return mlp_0_1_output == 77
        elif mlp_0_0_output in {32, 22, 79}:
            return mlp_0_1_output == 50
        elif mlp_0_0_output in {24, 76}:
            return mlp_0_1_output == 73
        elif mlp_0_0_output in {25, 68}:
            return mlp_0_1_output == 56
        elif mlp_0_0_output in {33, 26}:
            return mlp_0_1_output == 19
        elif mlp_0_0_output in {48, 27}:
            return mlp_0_1_output == 24
        elif mlp_0_0_output in {28, 53}:
            return mlp_0_1_output == 74
        elif mlp_0_0_output in {30}:
            return mlp_0_1_output == 39
        elif mlp_0_0_output in {31}:
            return mlp_0_1_output == 58
        elif mlp_0_0_output in {34}:
            return mlp_0_1_output == 78
        elif mlp_0_0_output in {36}:
            return mlp_0_1_output == 52
        elif mlp_0_0_output in {37}:
            return mlp_0_1_output == 5
        elif mlp_0_0_output in {39}:
            return mlp_0_1_output == 10
        elif mlp_0_0_output in {40}:
            return mlp_0_1_output == 12
        elif mlp_0_0_output in {41}:
            return mlp_0_1_output == 34
        elif mlp_0_0_output in {42}:
            return mlp_0_1_output == 2
        elif mlp_0_0_output in {43}:
            return mlp_0_1_output == 21
        elif mlp_0_0_output in {45}:
            return mlp_0_1_output == 69
        elif mlp_0_0_output in {46}:
            return mlp_0_1_output == 29
        elif mlp_0_0_output in {47}:
            return mlp_0_1_output == 31
        elif mlp_0_0_output in {49}:
            return mlp_0_1_output == 0
        elif mlp_0_0_output in {50}:
            return mlp_0_1_output == 37
        elif mlp_0_0_output in {52}:
            return mlp_0_1_output == 48
        elif mlp_0_0_output in {59, 54}:
            return mlp_0_1_output == 72
        elif mlp_0_0_output in {56}:
            return mlp_0_1_output == 16
        elif mlp_0_0_output in {58}:
            return mlp_0_1_output == 57
        elif mlp_0_0_output in {60}:
            return mlp_0_1_output == 18
        elif mlp_0_0_output in {72, 63}:
            return mlp_0_1_output == 28
        elif mlp_0_0_output in {69}:
            return mlp_0_1_output == 79
        elif mlp_0_0_output in {70}:
            return mlp_0_1_output == 62
        elif mlp_0_0_output in {77}:
            return mlp_0_1_output == 70

    num_attn_1_0_pattern = select(mlp_0_1_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_5_output, mlp_0_1_output):
        if attn_0_5_output in {"("}:
            return mlp_0_1_output == 38
        elif attn_0_5_output in {")"}:
            return mlp_0_1_output == 19
        elif attn_0_5_output in {"<s>"}:
            return mlp_0_1_output == 7

    num_attn_1_1_pattern = select(mlp_0_1_outputs, attn_0_5_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_4_output, mlp_0_1_output):
        if attn_0_4_output in {"("}:
            return mlp_0_1_output == 59
        elif attn_0_4_output in {")"}:
            return mlp_0_1_output == 17
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_1_output == 69

    num_attn_1_2_pattern = select(mlp_0_1_outputs, attn_0_4_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_0_output, mlp_0_0_output):
        if attn_0_0_output in {"<s>", ")", "("}:
            return mlp_0_0_output == 13

    num_attn_1_3_pattern = select(mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_5_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_2_output, mlp_0_0_output):
        if attn_0_2_output in {"("}:
            return mlp_0_0_output == 71
        elif attn_0_2_output in {")"}:
            return mlp_0_0_output == 13
        elif attn_0_2_output in {"<s>"}:
            return mlp_0_0_output == 61

    num_attn_1_4_pattern = select(mlp_0_0_outputs, attn_0_2_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_6_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_7_output, mlp_0_1_output):
        if attn_0_7_output in {"<s>", ")", "("}:
            return mlp_0_1_output == 55

    num_attn_1_5_pattern = select(mlp_0_1_outputs, attn_0_7_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_6_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, mlp_0_1_output):
        if position in {0, 71}:
            return mlp_0_1_output == 34
        elif position in {72, 1}:
            return mlp_0_1_output == 4
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
            34,
            35,
            36,
            38,
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
            52,
            53,
            54,
            55,
            56,
            57,
            59,
            60,
            62,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            73,
            75,
            76,
            77,
            78,
            79,
        }:
            return mlp_0_1_output == 71
        elif position in {33}:
            return mlp_0_1_output == 23
        elif position in {37}:
            return mlp_0_1_output == 52
        elif position in {39}:
            return mlp_0_1_output == 27
        elif position in {58, 46, 63}:
            return mlp_0_1_output == 45
        elif position in {49, 74, 61}:
            return mlp_0_1_output == 66

    num_attn_1_6_pattern = select(mlp_0_1_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_6_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(num_mlp_0_0_output, num_mlp_0_1_output):
        if num_mlp_0_0_output in {0, 72}:
            return num_mlp_0_1_output == 12
        elif num_mlp_0_0_output in {1}:
            return num_mlp_0_1_output == 9
        elif num_mlp_0_0_output in {17, 2, 51, 53}:
            return num_mlp_0_1_output == 61
        elif num_mlp_0_0_output in {66, 3, 6, 45, 60, 61}:
            return num_mlp_0_1_output == 26
        elif num_mlp_0_0_output in {4}:
            return num_mlp_0_1_output == 79
        elif num_mlp_0_0_output in {19, 5, 78}:
            return num_mlp_0_1_output == 59
        elif num_mlp_0_0_output in {75, 46, 7}:
            return num_mlp_0_1_output == 27
        elif num_mlp_0_0_output in {8}:
            return num_mlp_0_1_output == 6
        elif num_mlp_0_0_output in {9, 22}:
            return num_mlp_0_1_output == 34
        elif num_mlp_0_0_output in {10, 37, 31}:
            return num_mlp_0_1_output == 31
        elif num_mlp_0_0_output in {35, 11}:
            return num_mlp_0_1_output == 7
        elif num_mlp_0_0_output in {12}:
            return num_mlp_0_1_output == 76
        elif num_mlp_0_0_output in {65, 49, 13, 14}:
            return num_mlp_0_1_output == 64
        elif num_mlp_0_0_output in {15}:
            return num_mlp_0_1_output == 62
        elif num_mlp_0_0_output in {16}:
            return num_mlp_0_1_output == 15
        elif num_mlp_0_0_output in {18}:
            return num_mlp_0_1_output == 75
        elif num_mlp_0_0_output in {20}:
            return num_mlp_0_1_output == 54
        elif num_mlp_0_0_output in {44, 21}:
            return num_mlp_0_1_output == 1
        elif num_mlp_0_0_output in {77, 23}:
            return num_mlp_0_1_output == 23
        elif num_mlp_0_0_output in {24, 76}:
            return num_mlp_0_1_output == 18
        elif num_mlp_0_0_output in {25, 74}:
            return num_mlp_0_1_output == 52
        elif num_mlp_0_0_output in {26, 54}:
            return num_mlp_0_1_output == 0
        elif num_mlp_0_0_output in {48, 27}:
            return num_mlp_0_1_output == 77
        elif num_mlp_0_0_output in {28, 36}:
            return num_mlp_0_1_output == 24
        elif num_mlp_0_0_output in {29}:
            return num_mlp_0_1_output == 47
        elif num_mlp_0_0_output in {30}:
            return num_mlp_0_1_output == 53
        elif num_mlp_0_0_output in {32, 56}:
            return num_mlp_0_1_output == 40
        elif num_mlp_0_0_output in {33, 59, 68}:
            return num_mlp_0_1_output == 20
        elif num_mlp_0_0_output in {40, 34}:
            return num_mlp_0_1_output == 16
        elif num_mlp_0_0_output in {38}:
            return num_mlp_0_1_output == 32
        elif num_mlp_0_0_output in {64, 70, 39}:
            return num_mlp_0_1_output == 14
        elif num_mlp_0_0_output in {41}:
            return num_mlp_0_1_output == 74
        elif num_mlp_0_0_output in {42}:
            return num_mlp_0_1_output == 70
        elif num_mlp_0_0_output in {50, 43}:
            return num_mlp_0_1_output == 37
        elif num_mlp_0_0_output in {47}:
            return num_mlp_0_1_output == 28
        elif num_mlp_0_0_output in {52}:
            return num_mlp_0_1_output == 36
        elif num_mlp_0_0_output in {79, 55}:
            return num_mlp_0_1_output == 25
        elif num_mlp_0_0_output in {57}:
            return num_mlp_0_1_output == 66
        elif num_mlp_0_0_output in {58}:
            return num_mlp_0_1_output == 30
        elif num_mlp_0_0_output in {67, 62}:
            return num_mlp_0_1_output == 69
        elif num_mlp_0_0_output in {63}:
            return num_mlp_0_1_output == 58
        elif num_mlp_0_0_output in {69}:
            return num_mlp_0_1_output == 8
        elif num_mlp_0_0_output in {71}:
            return num_mlp_0_1_output == 10
        elif num_mlp_0_0_output in {73}:
            return num_mlp_0_1_output == 56

    num_attn_1_7_pattern = select(
        num_mlp_0_1_outputs, num_mlp_0_0_outputs, num_predicate_1_7
    )
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_7_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_6_output, attn_0_3_output):
        key = (attn_0_6_output, attn_0_3_output)
        if key in {("(", "("), ("(", "<s>"), ("<s>", "(")}:
            return 57
        elif key in {(")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 48
        return 56

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_6_output, attn_1_5_output):
        key = (attn_1_6_output, attn_1_5_output)
        return 2

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_1_5_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_5_output):
        key = num_attn_1_5_output
        return 63

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_5_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_5_output, num_attn_1_2_output):
        key = (num_attn_1_5_output, num_attn_1_2_output)
        return 70

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, mlp_0_1_output):
        if token in {")", "("}:
            return mlp_0_1_output == 71
        elif token in {"<s>"}:
            return mlp_0_1_output == 58

    attn_2_0_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_5_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_7_output, mlp_0_1_output):
        if attn_0_7_output in {")", "("}:
            return mlp_0_1_output == 71
        elif attn_0_7_output in {"<s>"}:
            return mlp_0_1_output == 5

    attn_2_1_pattern = select_closest(mlp_0_1_outputs, attn_0_7_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_4_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_7_output, position):
        if attn_0_7_output in {"("}:
            return position == 3
        elif attn_0_7_output in {")"}:
            return position == 7
        elif attn_0_7_output in {"<s>"}:
            return position == 2

    attn_2_2_pattern = select_closest(positions, attn_0_7_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, mlp_0_1_output):
        if token in {")", "("}:
            return mlp_0_1_output == 71
        elif token in {"<s>"}:
            return mlp_0_1_output == 5

    attn_2_3_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_7_output, mlp_0_1_output):
        if attn_0_7_output in {"<s>", ")", "("}:
            return mlp_0_1_output == 71

    attn_2_4_pattern = select_closest(mlp_0_1_outputs, attn_0_7_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_1_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_7_output, position):
        if attn_0_7_output in {"("}:
            return position == 4
        elif attn_0_7_output in {")"}:
            return position == 5
        elif attn_0_7_output in {"<s>"}:
            return position == 1

    attn_2_5_pattern = select_closest(positions, attn_0_7_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_2_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(token, mlp_0_1_output):
        if token in {")", "("}:
            return mlp_0_1_output == 71
        elif token in {"<s>"}:
            return mlp_0_1_output == 57

    attn_2_6_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_0_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_7_output, position):
        if attn_0_7_output in {"("}:
            return position == 2
        elif attn_0_7_output in {")"}:
            return position == 3
        elif attn_0_7_output in {"<s>"}:
            return position == 5

    attn_2_7_pattern = select_closest(positions, attn_0_7_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_2_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_1_1_output, mlp_0_1_output):
        if num_mlp_1_1_output in {
            0,
            1,
            6,
            7,
            9,
            14,
            21,
            25,
            28,
            31,
            33,
            35,
            36,
            38,
            41,
            42,
            43,
            48,
            49,
            52,
            53,
            56,
            58,
            59,
            61,
            63,
            64,
            66,
            68,
            70,
        }:
            return mlp_0_1_output == 66
        elif num_mlp_1_1_output in {2, 45}:
            return mlp_0_1_output == 73
        elif num_mlp_1_1_output in {
            65,
            3,
            67,
            5,
            8,
            10,
            75,
            12,
            76,
            79,
            16,
            19,
            22,
            24,
            26,
            30,
        }:
            return mlp_0_1_output == 45
        elif num_mlp_1_1_output in {
            32,
            4,
            71,
            40,
            74,
            11,
            44,
            77,
            78,
            47,
            18,
            50,
            20,
            55,
            57,
            27,
            29,
            62,
        }:
            return mlp_0_1_output == 4
        elif num_mlp_1_1_output in {13}:
            return mlp_0_1_output == 58
        elif num_mlp_1_1_output in {15}:
            return mlp_0_1_output == 32
        elif num_mlp_1_1_output in {17}:
            return mlp_0_1_output == 47
        elif num_mlp_1_1_output in {23}:
            return mlp_0_1_output == 71
        elif num_mlp_1_1_output in {34}:
            return mlp_0_1_output == 56
        elif num_mlp_1_1_output in {73, 37}:
            return mlp_0_1_output == 2
        elif num_mlp_1_1_output in {39}:
            return mlp_0_1_output == 57
        elif num_mlp_1_1_output in {46}:
            return mlp_0_1_output == 10
        elif num_mlp_1_1_output in {51}:
            return mlp_0_1_output == 9
        elif num_mlp_1_1_output in {54}:
            return mlp_0_1_output == 55
        elif num_mlp_1_1_output in {60}:
            return mlp_0_1_output == 61
        elif num_mlp_1_1_output in {69}:
            return mlp_0_1_output == 75
        elif num_mlp_1_1_output in {72}:
            return mlp_0_1_output == 40

    num_attn_2_0_pattern = select(
        mlp_0_1_outputs, num_mlp_1_1_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_6_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_3_output, attn_0_0_output):
        if attn_1_3_output in {
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
            return attn_0_0_output == ""
        elif attn_1_3_output in {33}:
            return attn_0_0_output == "<s>"
        elif attn_1_3_output in {70}:
            return attn_0_0_output == "<pad>"

    num_attn_2_1_pattern = select(attn_0_0_outputs, attn_1_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_7_output, mlp_1_1_output):
        if attn_1_7_output in {"("}:
            return mlp_1_1_output == 36
        elif attn_1_7_output in {")"}:
            return mlp_1_1_output == 35
        elif attn_1_7_output in {"<s>"}:
            return mlp_1_1_output == 76

    num_attn_2_2_pattern = select(mlp_1_1_outputs, attn_1_7_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_1_0_output, attn_1_3_output):
        if mlp_1_0_output in {0, 12}:
            return attn_1_3_output == 65
        elif mlp_1_0_output in {
            1,
            3,
            8,
            9,
            10,
            11,
            14,
            19,
            20,
            25,
            27,
            32,
            33,
            34,
            35,
            37,
            38,
            41,
            44,
            45,
            47,
            48,
            50,
            52,
            54,
            55,
            56,
            57,
            61,
            64,
            66,
            73,
            74,
            75,
            79,
        }:
            return attn_1_3_output == 71
        elif mlp_1_0_output in {
            2,
            6,
            7,
            13,
            16,
            17,
            18,
            22,
            23,
            24,
            26,
            30,
            42,
            46,
            51,
            53,
            65,
            68,
            69,
            70,
            71,
            77,
        }:
            return attn_1_3_output == 13
        elif mlp_1_0_output in {4}:
            return attn_1_3_output == 59
        elif mlp_1_0_output in {5}:
            return attn_1_3_output == 2
        elif mlp_1_0_output in {31, 36, 15}:
            return attn_1_3_output == 47
        elif mlp_1_0_output in {21}:
            return attn_1_3_output == 39
        elif mlp_1_0_output in {58, 28}:
            return attn_1_3_output == 49
        elif mlp_1_0_output in {29}:
            return attn_1_3_output == 7
        elif mlp_1_0_output in {39}:
            return attn_1_3_output == 28
        elif mlp_1_0_output in {40, 76}:
            return attn_1_3_output == 57
        elif mlp_1_0_output in {43}:
            return attn_1_3_output == 45
        elif mlp_1_0_output in {49}:
            return attn_1_3_output == 41
        elif mlp_1_0_output in {72, 59}:
            return attn_1_3_output == 51
        elif mlp_1_0_output in {60}:
            return attn_1_3_output == 33
        elif mlp_1_0_output in {62}:
            return attn_1_3_output == 17
        elif mlp_1_0_output in {63}:
            return attn_1_3_output == 31
        elif mlp_1_0_output in {67}:
            return attn_1_3_output == 69
        elif mlp_1_0_output in {78}:
            return attn_1_3_output == 22

    num_attn_2_3_pattern = select(attn_1_3_outputs, mlp_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_5_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(q_mlp_1_0_output, k_mlp_1_0_output):
        if q_mlp_1_0_output in {0}:
            return k_mlp_1_0_output == 7
        elif q_mlp_1_0_output in {1}:
            return k_mlp_1_0_output == 55
        elif q_mlp_1_0_output in {2, 13, 14, 17, 26, 30}:
            return k_mlp_1_0_output == 35
        elif q_mlp_1_0_output in {66, 3, 68, 5, 40, 15, 49, 62, 31}:
            return k_mlp_1_0_output == 31
        elif q_mlp_1_0_output in {4}:
            return k_mlp_1_0_output == 2
        elif q_mlp_1_0_output in {6}:
            return k_mlp_1_0_output == 28
        elif q_mlp_1_0_output in {7}:
            return k_mlp_1_0_output == 43
        elif q_mlp_1_0_output in {32, 8, 10, 11, 74, 47, 18, 52, 53}:
            return k_mlp_1_0_output == 25
        elif q_mlp_1_0_output in {9}:
            return k_mlp_1_0_output == 1
        elif q_mlp_1_0_output in {12}:
            return k_mlp_1_0_output == 36
        elif q_mlp_1_0_output in {16}:
            return k_mlp_1_0_output == 23
        elif q_mlp_1_0_output in {19}:
            return k_mlp_1_0_output == 67
        elif q_mlp_1_0_output in {24, 41, 20}:
            return k_mlp_1_0_output == 64
        elif q_mlp_1_0_output in {21}:
            return k_mlp_1_0_output == 38
        elif q_mlp_1_0_output in {22}:
            return k_mlp_1_0_output == 75
        elif q_mlp_1_0_output in {23}:
            return k_mlp_1_0_output == 29
        elif q_mlp_1_0_output in {25}:
            return k_mlp_1_0_output == 12
        elif q_mlp_1_0_output in {27}:
            return k_mlp_1_0_output == 10
        elif q_mlp_1_0_output in {72, 28, 78}:
            return k_mlp_1_0_output == 56
        elif q_mlp_1_0_output in {29}:
            return k_mlp_1_0_output == 4
        elif q_mlp_1_0_output in {33, 75}:
            return k_mlp_1_0_output == 58
        elif q_mlp_1_0_output in {34, 63}:
            return k_mlp_1_0_output == 52
        elif q_mlp_1_0_output in {35, 36, 76}:
            return k_mlp_1_0_output == 47
        elif q_mlp_1_0_output in {37, 71}:
            return k_mlp_1_0_output == 76
        elif q_mlp_1_0_output in {70, 60, 38}:
            return k_mlp_1_0_output == 19
        elif q_mlp_1_0_output in {39}:
            return k_mlp_1_0_output == 61
        elif q_mlp_1_0_output in {42}:
            return k_mlp_1_0_output == 54
        elif q_mlp_1_0_output in {43}:
            return k_mlp_1_0_output == 50
        elif q_mlp_1_0_output in {44}:
            return k_mlp_1_0_output == 14
        elif q_mlp_1_0_output in {45}:
            return k_mlp_1_0_output == 59
        elif q_mlp_1_0_output in {46}:
            return k_mlp_1_0_output == 6
        elif q_mlp_1_0_output in {48, 65, 73}:
            return k_mlp_1_0_output == 41
        elif q_mlp_1_0_output in {50}:
            return k_mlp_1_0_output == 11
        elif q_mlp_1_0_output in {51}:
            return k_mlp_1_0_output == 5
        elif q_mlp_1_0_output in {54}:
            return k_mlp_1_0_output == 72
        elif q_mlp_1_0_output in {64, 69, 55, 58, 59}:
            return k_mlp_1_0_output == 37
        elif q_mlp_1_0_output in {56, 61}:
            return k_mlp_1_0_output == 63
        elif q_mlp_1_0_output in {57}:
            return k_mlp_1_0_output == 42
        elif q_mlp_1_0_output in {67}:
            return k_mlp_1_0_output == 60
        elif q_mlp_1_0_output in {77}:
            return k_mlp_1_0_output == 24
        elif q_mlp_1_0_output in {79}:
            return k_mlp_1_0_output == 21

    num_attn_2_4_pattern = select(mlp_1_0_outputs, mlp_1_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_4_output, token):
        if attn_1_4_output in {"<s>", ")", "("}:
            return token == ""

    num_attn_2_5_pattern = select(tokens, attn_1_4_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_2_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_6_output, attn_1_1_output):
        if attn_1_6_output in {"<s>", ")", "("}:
            return attn_1_1_output == ""

    num_attn_2_6_pattern = select(attn_1_1_outputs, attn_1_6_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_4_output, attn_0_1_output):
        if attn_1_4_output in {"<s>", ")", "("}:
            return attn_0_1_output == ""

    num_attn_2_7_pattern = select(attn_0_1_outputs, attn_1_4_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_4_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_2_4_output):
        key = (attn_2_3_output, attn_2_4_output)
        if key in {
            ("(", ")"),
            ("(", "<s>"),
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            ("<s>", ")"),
        }:
            return 40
        return 61

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_4_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_5_output, attn_2_7_output):
        key = (attn_2_5_output, attn_2_7_output)
        if key in {("(", "("), ("(", ")"), ("(", "<s>"), ("<s>", "(")}:
            return 43
        elif key in {(")", "(")}:
            return 62
        return 79

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_5_outputs, attn_2_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_2_output, num_attn_1_5_output):
        key = (num_attn_2_2_output, num_attn_1_5_output)
        return 40

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_4_output, num_attn_1_0_output):
        key = (num_attn_2_4_output, num_attn_1_0_output)
        return 26

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_4_outputs, num_attn_1_0_outputs)
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
            "(",
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
            "(",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            "(",
            ")",
            "(",
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            "(",
            ")",
            ")",
            "(",
            ")",
            "(",
            ")",
            "(",
        ]
    )
)
