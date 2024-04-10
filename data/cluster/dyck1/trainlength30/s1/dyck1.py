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
        "output/length/rasp/dyck1/trainlength30/s1/dyck1_weights.csv",
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
    def predicate_0_0(token, position):
        if token in {"("}:
            return position == 7
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 44

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"("}:
            return position == 9
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 2

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 34, 4, 49, 51, 58, 59}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 58
        elif q_position in {2}:
            return k_position == 26
        elif q_position in {25, 3}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 52
        elif q_position in {32, 36, 6, 44, 31}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 28
        elif q_position in {8, 17, 23}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 43
        elif q_position in {10, 11, 28, 13}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {18, 19, 15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {40, 57, 20, 22}:
            return k_position == 7
        elif q_position in {21}:
            return k_position == 12
        elif q_position in {24, 26}:
            return k_position == 8
        elif q_position in {27, 37}:
            return k_position == 39
        elif q_position in {33, 29}:
            return k_position == 29
        elif q_position in {38, 48, 30}:
            return k_position == 44
        elif q_position in {56, 35}:
            return k_position == 30
        elif q_position in {39}:
            return k_position == 33
        elif q_position in {41}:
            return k_position == 27
        elif q_position in {42, 46}:
            return k_position == 54
        elif q_position in {43, 47}:
            return k_position == 34
        elif q_position in {45}:
            return k_position == 53
        elif q_position in {50}:
            return k_position == 40
        elif q_position in {52}:
            return k_position == 1
        elif q_position in {53}:
            return k_position == 31
        elif q_position in {54}:
            return k_position == 55
        elif q_position in {55}:
            return k_position == 50

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 44
        elif q_position in {1}:
            return k_position == 29
        elif q_position in {2}:
            return k_position == 24
        elif q_position in {3, 29}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 28
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {27, 7}:
            return k_position == 4
        elif q_position in {8, 20}:
            return k_position == 6
        elif q_position in {36, 37, 9, 44, 52, 22, 26}:
            return k_position == 7
        elif q_position in {24, 10, 11, 14}:
            return k_position == 9
        elif q_position in {12, 23, 15}:
            return k_position == 11
        elif q_position in {28, 13}:
            return k_position == 10
        elif q_position in {16}:
            return k_position == 14
        elif q_position in {17, 54}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 12
        elif q_position in {42, 19}:
            return k_position == 16
        elif q_position in {40, 21}:
            return k_position == 17
        elif q_position in {25}:
            return k_position == 8
        elif q_position in {30}:
            return k_position == 48
        elif q_position in {31}:
            return k_position == 56
        elif q_position in {32, 55}:
            return k_position == 25
        elif q_position in {33, 38}:
            return k_position == 39
        elif q_position in {34}:
            return k_position == 41
        elif q_position in {51, 35}:
            return k_position == 55
        elif q_position in {57, 43, 39}:
            return k_position == 33
        elif q_position in {41}:
            return k_position == 43
        elif q_position in {56, 45}:
            return k_position == 57
        elif q_position in {58, 46}:
            return k_position == 32
        elif q_position in {47}:
            return k_position == 40
        elif q_position in {48}:
            return k_position == 35
        elif q_position in {49}:
            return k_position == 13
        elif q_position in {50}:
            return k_position == 45
        elif q_position in {53}:
            return k_position == 21
        elif q_position in {59}:
            return k_position == 36

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 48}:
            return k_position == 45
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {8, 2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {43, 7}:
            return k_position == 36
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {34, 36, 10, 47, 56}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {50, 14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {32, 20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {52, 22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 7
        elif q_position in {25, 30}:
            return k_position == 24
        elif q_position in {26, 54}:
            return k_position == 25
        elif q_position in {41, 27}:
            return k_position == 26
        elif q_position in {28, 44}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {57, 31}:
            return k_position == 51
        elif q_position in {33, 53, 37}:
            return k_position == 43
        elif q_position in {35}:
            return k_position == 53
        elif q_position in {38, 55}:
            return k_position == 40
        elif q_position in {39}:
            return k_position == 31
        elif q_position in {40}:
            return k_position == 30
        elif q_position in {42}:
            return k_position == 50
        elif q_position in {45}:
            return k_position == 52
        elif q_position in {46}:
            return k_position == 49
        elif q_position in {49}:
            return k_position == 56
        elif q_position in {59, 58, 51}:
            return k_position == 41

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {8, 1, 57}:
            return k_position == 7
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3, 5}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 29
        elif q_position in {9}:
            return k_position == 53
        elif q_position in {10, 11}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {49, 13}:
            return k_position == 39
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {44, 53, 15}:
            return k_position == 35
        elif q_position in {16}:
            return k_position == 14
        elif q_position in {17, 46, 31}:
            return k_position == 48
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 16
        elif q_position in {20}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 19
        elif q_position in {26, 22}:
            return k_position == 9
        elif q_position in {23}:
            return k_position == 21
        elif q_position in {24}:
            return k_position == 22
        elif q_position in {25, 45}:
            return k_position == 31
        elif q_position in {43, 27}:
            return k_position == 25
        elif q_position in {28}:
            return k_position == 24
        elif q_position in {40, 29}:
            return k_position == 27
        elif q_position in {30}:
            return k_position == 13
        elif q_position in {32, 33}:
            return k_position == 30
        elif q_position in {34}:
            return k_position == 36
        elif q_position in {41, 35}:
            return k_position == 57
        elif q_position in {36, 37}:
            return k_position == 59
        elif q_position in {42, 38}:
            return k_position == 46
        elif q_position in {52, 39}:
            return k_position == 44
        elif q_position in {55, 47}:
            return k_position == 49
        elif q_position in {48, 51}:
            return k_position == 33
        elif q_position in {50}:
            return k_position == 32
        elif q_position in {54}:
            return k_position == 43
        elif q_position in {56}:
            return k_position == 38
        elif q_position in {58}:
            return k_position == 52
        elif q_position in {59}:
            return k_position == 55

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 27

    attn_0_6_pattern = select_closest(positions, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 3

    attn_0_7_pattern = select_closest(positions, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 13
        elif q_position in {1, 10, 12}:
            return k_position == 0
        elif q_position in {24, 41, 2}:
            return k_position == 46
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 14
        elif q_position in {13, 5, 47}:
            return k_position == 7
        elif q_position in {6, 31}:
            return k_position == 30
        elif q_position in {7}:
            return k_position == 21
        elif q_position in {8, 48, 27, 46}:
            return k_position == 52
        elif q_position in {9}:
            return k_position == 54
        elif q_position in {51, 26, 11}:
            return k_position == 58
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {54, 15}:
            return k_position == 5
        elif q_position in {16}:
            return k_position == 10
        elif q_position in {17}:
            return k_position == 47
        elif q_position in {18}:
            return k_position == 20
        elif q_position in {19, 38}:
            return k_position == 57
        elif q_position in {34, 20}:
            return k_position == 48
        elif q_position in {21, 30}:
            return k_position == 36
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 34
        elif q_position in {25, 28, 33}:
            return k_position == 35
        elif q_position in {29}:
            return k_position == 56
        elif q_position in {32, 44}:
            return k_position == 55
        elif q_position in {56, 58, 35, 53}:
            return k_position == 1
        elif q_position in {36}:
            return k_position == 29
        elif q_position in {37}:
            return k_position == 59
        elif q_position in {39}:
            return k_position == 33
        elif q_position in {40}:
            return k_position == 53
        elif q_position in {42}:
            return k_position == 40
        elif q_position in {57, 43}:
            return k_position == 42
        elif q_position in {45}:
            return k_position == 37
        elif q_position in {49}:
            return k_position == 41
        elif q_position in {50}:
            return k_position == 6
        elif q_position in {52}:
            return k_position == 50
        elif q_position in {55}:
            return k_position == 43
        elif q_position in {59}:
            return k_position == 51

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 28
        elif q_position in {2, 54}:
            return k_position == 8
        elif q_position in {40, 3}:
            return k_position == 54
        elif q_position in {4, 21, 7}:
            return k_position == 35
        elif q_position in {5}:
            return k_position == 21
        elif q_position in {52, 13, 6}:
            return k_position == 52
        elif q_position in {8, 25, 10}:
            return k_position == 50
        elif q_position in {9}:
            return k_position == 44
        elif q_position in {11}:
            return k_position == 48
        elif q_position in {48, 57, 12}:
            return k_position == 11
        elif q_position in {19, 14}:
            return k_position == 32
        elif q_position in {46, 15}:
            return k_position == 47
        elif q_position in {16, 24}:
            return k_position == 38
        elif q_position in {17}:
            return k_position == 53
        elif q_position in {18, 38}:
            return k_position == 36
        elif q_position in {20}:
            return k_position == 37
        elif q_position in {33, 26, 28, 22}:
            return k_position == 39
        elif q_position in {23}:
            return k_position == 33
        elif q_position in {27}:
            return k_position == 58
        elif q_position in {29}:
            return k_position == 55
        elif q_position in {56, 42, 50, 30}:
            return k_position == 1
        elif q_position in {31}:
            return k_position == 17
        elif q_position in {32}:
            return k_position == 12
        elif q_position in {34, 36, 45}:
            return k_position == 5
        elif q_position in {35}:
            return k_position == 9
        elif q_position in {37}:
            return k_position == 15
        elif q_position in {39}:
            return k_position == 14
        elif q_position in {41}:
            return k_position == 26
        elif q_position in {58, 43}:
            return k_position == 31
        elif q_position in {59, 44}:
            return k_position == 29
        elif q_position in {47}:
            return k_position == 7
        elif q_position in {49, 51}:
            return k_position == 43
        elif q_position in {53}:
            return k_position == 10
        elif q_position in {55}:
            return k_position == 59

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 10
        elif token in {")"}:
            return position == 45
        elif token in {"<s>"}:
            return position == 37

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 19
        elif q_position in {1, 9}:
            return k_position == 27
        elif q_position in {2}:
            return k_position == 53
        elif q_position in {3}:
            return k_position == 40
        elif q_position in {34, 4}:
            return k_position == 7
        elif q_position in {18, 5, 22}:
            return k_position == 59
        elif q_position in {42, 6}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 33
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {10, 43}:
            return k_position == 29
        elif q_position in {11}:
            return k_position == 25
        elif q_position in {33, 40, 12, 19, 29}:
            return k_position == 37
        elif q_position in {13}:
            return k_position == 52
        elif q_position in {14}:
            return k_position == 32
        elif q_position in {15}:
            return k_position == 5
        elif q_position in {16}:
            return k_position == 24
        elif q_position in {17}:
            return k_position == 34
        elif q_position in {20, 44}:
            return k_position == 55
        elif q_position in {41, 21}:
            return k_position == 58
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24, 31}:
            return k_position == 35
        elif q_position in {25, 54}:
            return k_position == 56
        elif q_position in {26, 28, 47}:
            return k_position == 39
        elif q_position in {48, 27}:
            return k_position == 31
        elif q_position in {30}:
            return k_position == 11
        elif q_position in {32, 35, 37, 39, 45, 51, 56}:
            return k_position == 1
        elif q_position in {36}:
            return k_position == 47
        elif q_position in {38}:
            return k_position == 57
        elif q_position in {46}:
            return k_position == 30
        elif q_position in {49}:
            return k_position == 36
        elif q_position in {50, 53}:
            return k_position == 14
        elif q_position in {59, 52}:
            return k_position == 41
        elif q_position in {55}:
            return k_position == 6
        elif q_position in {57}:
            return k_position == 51
        elif q_position in {58}:
            return k_position == 44

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"("}:
            return position == 47
        elif token in {")"}:
            return position == 0
        elif token in {"<s>"}:
            return position == 15

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0, 1, 4}:
            return k_position == 52
        elif q_position in {2, 51}:
            return k_position == 56
        elif q_position in {3}:
            return k_position == 13
        elif q_position in {48, 5}:
            return k_position == 38
        elif q_position in {57, 43, 6}:
            return k_position == 27
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 44
        elif q_position in {9, 41, 30, 39}:
            return k_position == 39
        elif q_position in {10, 59, 21}:
            return k_position == 33
        elif q_position in {16, 42, 11, 52}:
            return k_position == 26
        elif q_position in {32, 12, 45, 17, 31}:
            return k_position == 30
        elif q_position in {13, 54}:
            return k_position == 36
        elif q_position in {14}:
            return k_position == 18
        elif q_position in {37, 15}:
            return k_position == 34
        elif q_position in {18}:
            return k_position == 23
        elif q_position in {19}:
            return k_position == 58
        elif q_position in {20}:
            return k_position == 41
        elif q_position in {35, 22}:
            return k_position == 37
        elif q_position in {23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 51
        elif q_position in {25}:
            return k_position == 46
        elif q_position in {56, 26}:
            return k_position == 59
        elif q_position in {27}:
            return k_position == 28
        elif q_position in {34, 28}:
            return k_position == 48
        elif q_position in {50, 29}:
            return k_position == 50
        elif q_position in {33}:
            return k_position == 45
        elif q_position in {36}:
            return k_position == 35
        elif q_position in {40, 38}:
            return k_position == 57
        elif q_position in {44}:
            return k_position == 29
        elif q_position in {53, 46}:
            return k_position == 31
        elif q_position in {47}:
            return k_position == 32
        elif q_position in {49, 55}:
            return k_position == 47
        elif q_position in {58}:
            return k_position == 49

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {
            0,
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
        elif position in {2, 7, 9, 11, 13}:
            return token == "<s>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {
            0,
            2,
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
        elif position in {3, 4, 5, 7}:
            return token == ")"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        if key in {("(", "("), ("(", "<s>")}:
            return 1
        return 33

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, attn_0_4_output):
        key = (attn_0_3_output, attn_0_4_output)
        if key in {("(", "("), ("(", "<s>"), (")", "("), ("<s>", "("), ("<s>", "<s>")}:
            return 39
        elif key in {(")", ")"), ("<s>", ")")}:
            return 2
        return 52

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_4_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_6_output):
        key = (num_attn_0_1_output, num_attn_0_6_output)
        return 12

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        return 12

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_1_output, position):
        if mlp_0_1_output in {0}:
            return position == 35
        elif mlp_0_1_output in {24, 1}:
            return position == 52
        elif mlp_0_1_output in {2, 51, 34, 52}:
            return position == 11
        elif mlp_0_1_output in {3}:
            return position == 5
        elif mlp_0_1_output in {4}:
            return position == 43
        elif mlp_0_1_output in {5, 9, 44, 20, 23}:
            return position == 27
        elif mlp_0_1_output in {6, 11, 12, 14, 17, 22, 55, 27, 31}:
            return position == 23
        elif mlp_0_1_output in {7, 8, 45, 46, 16, 50}:
            return position == 21
        elif mlp_0_1_output in {10, 47, 18, 21, 28}:
            return position == 25
        elif mlp_0_1_output in {35, 13}:
            return position == 36
        elif mlp_0_1_output in {15}:
            return position == 34
        elif mlp_0_1_output in {19}:
            return position == 56
        elif mlp_0_1_output in {25}:
            return position == 59
        elif mlp_0_1_output in {26}:
            return position == 22
        elif mlp_0_1_output in {29}:
            return position == 33
        elif mlp_0_1_output in {54, 30}:
            return position == 13
        elif mlp_0_1_output in {32, 41}:
            return position == 41
        elif mlp_0_1_output in {33}:
            return position == 49
        elif mlp_0_1_output in {36}:
            return position == 18
        elif mlp_0_1_output in {37}:
            return position == 3
        elif mlp_0_1_output in {42, 53, 38}:
            return position == 19
        elif mlp_0_1_output in {39}:
            return position == 2
        elif mlp_0_1_output in {40, 57, 48}:
            return position == 16
        elif mlp_0_1_output in {56, 43}:
            return position == 17
        elif mlp_0_1_output in {49}:
            return position == 47
        elif mlp_0_1_output in {58}:
            return position == 6
        elif mlp_0_1_output in {59}:
            return position == 14

    attn_1_0_pattern = select_closest(positions, mlp_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_7_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_3_output, position):
        if attn_0_3_output in {"<s>", "("}:
            return position == 1
        elif attn_0_3_output in {")"}:
            return position == 5

    attn_1_1_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 41

    attn_1_2_pattern = select_closest(positions, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"("}:
            return position == 3
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 1

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, position):
        if token in {"<s>", "("}:
            return position == 1
        elif token in {")"}:
            return position == 8

    attn_1_4_pattern = select_closest(positions, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 1, 32, 3, 35, 37, 41, 48, 17, 49, 55}:
            return position == 1
        elif mlp_0_1_output in {2}:
            return position == 3
        elif mlp_0_1_output in {4}:
            return position == 45
        elif mlp_0_1_output in {5}:
            return position == 33
        elif mlp_0_1_output in {38, 57, 6, 22}:
            return position == 25
        elif mlp_0_1_output in {47, 7}:
            return position == 54
        elif mlp_0_1_output in {8, 33}:
            return position == 38
        elif mlp_0_1_output in {9}:
            return position == 49
        elif mlp_0_1_output in {25, 10, 12}:
            return position == 27
        elif mlp_0_1_output in {11}:
            return position == 29
        elif mlp_0_1_output in {13}:
            return position == 31
        elif mlp_0_1_output in {14}:
            return position == 41
        elif mlp_0_1_output in {15}:
            return position == 42
        elif mlp_0_1_output in {16}:
            return position == 32
        elif mlp_0_1_output in {18, 36}:
            return position == 35
        elif mlp_0_1_output in {19, 23}:
            return position == 51
        elif mlp_0_1_output in {20}:
            return position == 56
        elif mlp_0_1_output in {21}:
            return position == 43
        elif mlp_0_1_output in {39, 50, 24, 59, 29}:
            return position == 4
        elif mlp_0_1_output in {26}:
            return position == 15
        elif mlp_0_1_output in {42, 27}:
            return position == 36
        elif mlp_0_1_output in {28}:
            return position == 24
        elif mlp_0_1_output in {30}:
            return position == 22
        elif mlp_0_1_output in {31}:
            return position == 23
        elif mlp_0_1_output in {34}:
            return position == 19
        elif mlp_0_1_output in {40}:
            return position == 8
        elif mlp_0_1_output in {43}:
            return position == 52
        elif mlp_0_1_output in {44, 46}:
            return position == 40
        elif mlp_0_1_output in {45}:
            return position == 5
        elif mlp_0_1_output in {51}:
            return position == 6
        elif mlp_0_1_output in {52}:
            return position == 9
        elif mlp_0_1_output in {53}:
            return position == 14
        elif mlp_0_1_output in {54}:
            return position == 50
        elif mlp_0_1_output in {56}:
            return position == 20
        elif mlp_0_1_output in {58}:
            return position == 44

    attn_1_5_pattern = select_closest(positions, mlp_0_1_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_4_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 3, 36, 7, 9, 10, 11, 14, 17, 18, 50, 51, 25, 27}:
            return k_position == 5
        elif q_position in {1, 21, 39}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 24
        elif q_position in {4, 6, 8, 43, 12, 16, 19, 23, 29}:
            return k_position == 3
        elif q_position in {33, 5}:
            return k_position == 53
        elif q_position in {13}:
            return k_position == 31
        elif q_position in {15}:
            return k_position == 10
        elif q_position in {20, 53}:
            return k_position == 39
        elif q_position in {22}:
            return k_position == 8
        elif q_position in {24, 57, 38}:
            return k_position == 9
        elif q_position in {26}:
            return k_position == 44
        elif q_position in {28}:
            return k_position == 23
        elif q_position in {59, 30}:
            return k_position == 56
        elif q_position in {48, 31}:
            return k_position == 37
        elif q_position in {32}:
            return k_position == 35
        elif q_position in {34}:
            return k_position == 25
        elif q_position in {56, 35}:
            return k_position == 16
        elif q_position in {37}:
            return k_position == 40
        elif q_position in {40}:
            return k_position == 33
        elif q_position in {41}:
            return k_position == 50
        elif q_position in {42, 45, 54}:
            return k_position == 11
        elif q_position in {44}:
            return k_position == 55
        elif q_position in {46}:
            return k_position == 7
        elif q_position in {47}:
            return k_position == 41
        elif q_position in {49}:
            return k_position == 51
        elif q_position in {52, 55}:
            return k_position == 36
        elif q_position in {58}:
            return k_position == 58

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 56

    attn_1_7_pattern = select_closest(positions, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_2_output, mlp_0_1_output):
        if attn_0_2_output in {")", "("}:
            return mlp_0_1_output == 39
        elif attn_0_2_output in {"<s>"}:
            return mlp_0_1_output == 24

    num_attn_1_0_pattern = select(mlp_0_1_outputs, attn_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0, 36}:
            return k_mlp_0_1_output == 45
        elif q_mlp_0_1_output in {1, 5, 6}:
            return k_mlp_0_1_output == 10
        elif q_mlp_0_1_output in {2, 44}:
            return k_mlp_0_1_output == 23
        elif q_mlp_0_1_output in {3, 47}:
            return k_mlp_0_1_output == 24
        elif q_mlp_0_1_output in {4}:
            return k_mlp_0_1_output == 20
        elif q_mlp_0_1_output in {56, 19, 7}:
            return k_mlp_0_1_output == 41
        elif q_mlp_0_1_output in {8, 55}:
            return k_mlp_0_1_output == 15
        elif q_mlp_0_1_output in {9, 28}:
            return k_mlp_0_1_output == 36
        elif q_mlp_0_1_output in {33, 10, 38}:
            return k_mlp_0_1_output == 54
        elif q_mlp_0_1_output in {11}:
            return k_mlp_0_1_output == 12
        elif q_mlp_0_1_output in {58, 12, 22}:
            return k_mlp_0_1_output == 9
        elif q_mlp_0_1_output in {13}:
            return k_mlp_0_1_output == 51
        elif q_mlp_0_1_output in {14}:
            return k_mlp_0_1_output == 8
        elif q_mlp_0_1_output in {15}:
            return k_mlp_0_1_output == 29
        elif q_mlp_0_1_output in {16}:
            return k_mlp_0_1_output == 30
        elif q_mlp_0_1_output in {24, 17}:
            return k_mlp_0_1_output == 16
        elif q_mlp_0_1_output in {49, 18}:
            return k_mlp_0_1_output == 14
        elif q_mlp_0_1_output in {20}:
            return k_mlp_0_1_output == 37
        elif q_mlp_0_1_output in {21}:
            return k_mlp_0_1_output == 6
        elif q_mlp_0_1_output in {34, 23}:
            return k_mlp_0_1_output == 25
        elif q_mlp_0_1_output in {25}:
            return k_mlp_0_1_output == 40
        elif q_mlp_0_1_output in {26, 37}:
            return k_mlp_0_1_output == 33
        elif q_mlp_0_1_output in {27}:
            return k_mlp_0_1_output == 21
        elif q_mlp_0_1_output in {29, 30}:
            return k_mlp_0_1_output == 28
        elif q_mlp_0_1_output in {31}:
            return k_mlp_0_1_output == 38
        elif q_mlp_0_1_output in {32}:
            return k_mlp_0_1_output == 19
        elif q_mlp_0_1_output in {48, 35}:
            return k_mlp_0_1_output == 13
        elif q_mlp_0_1_output in {41, 39}:
            return k_mlp_0_1_output == 22
        elif q_mlp_0_1_output in {40}:
            return k_mlp_0_1_output == 49
        elif q_mlp_0_1_output in {42, 59}:
            return k_mlp_0_1_output == 26
        elif q_mlp_0_1_output in {43}:
            return k_mlp_0_1_output == 35
        elif q_mlp_0_1_output in {51, 45}:
            return k_mlp_0_1_output == 57
        elif q_mlp_0_1_output in {46}:
            return k_mlp_0_1_output == 17
        elif q_mlp_0_1_output in {50, 52}:
            return k_mlp_0_1_output == 48
        elif q_mlp_0_1_output in {53}:
            return k_mlp_0_1_output == 47
        elif q_mlp_0_1_output in {54}:
            return k_mlp_0_1_output == 31
        elif q_mlp_0_1_output in {57}:
            return k_mlp_0_1_output == 55

    num_attn_1_1_pattern = select(mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_4_output, mlp_0_1_output):
        if attn_0_4_output in {"("}:
            return mlp_0_1_output == 51
        elif attn_0_4_output in {")"}:
            return mlp_0_1_output == 53
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_1_output == 2

    num_attn_1_2_pattern = select(mlp_0_1_outputs, attn_0_4_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_7_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0}:
            return k_mlp_0_1_output == 15
        elif q_mlp_0_1_output in {
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            11,
            14,
            15,
            16,
            21,
            23,
            24,
            27,
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
            45,
            46,
            47,
            49,
            51,
            52,
            53,
            54,
            55,
            57,
            58,
            59,
        }:
            return k_mlp_0_1_output == 2
        elif q_mlp_0_1_output in {4}:
            return k_mlp_0_1_output == 59
        elif q_mlp_0_1_output in {32, 9, 10, 13, 17, 18, 25}:
            return k_mlp_0_1_output == 52
        elif q_mlp_0_1_output in {19, 12, 20}:
            return k_mlp_0_1_output == 51
        elif q_mlp_0_1_output in {22}:
            return k_mlp_0_1_output == 33
        elif q_mlp_0_1_output in {26}:
            return k_mlp_0_1_output == 31
        elif q_mlp_0_1_output in {28}:
            return k_mlp_0_1_output == 21
        elif q_mlp_0_1_output in {29}:
            return k_mlp_0_1_output == 20
        elif q_mlp_0_1_output in {30}:
            return k_mlp_0_1_output == 16
        elif q_mlp_0_1_output in {31}:
            return k_mlp_0_1_output == 5
        elif q_mlp_0_1_output in {44}:
            return k_mlp_0_1_output == 58
        elif q_mlp_0_1_output in {48}:
            return k_mlp_0_1_output == 7
        elif q_mlp_0_1_output in {50}:
            return k_mlp_0_1_output == 18
        elif q_mlp_0_1_output in {56}:
            return k_mlp_0_1_output == 46

    num_attn_1_3_pattern = select(mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_4_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {
            0,
            3,
            5,
            7,
            8,
            11,
            12,
            16,
            18,
            20,
            21,
            22,
            27,
            31,
            33,
            34,
            35,
            42,
            46,
            48,
            51,
            53,
            56,
            59,
        }:
            return k_mlp_0_0_output == 1
        elif q_mlp_0_0_output in {1}:
            return k_mlp_0_0_output == 14
        elif q_mlp_0_0_output in {2, 40, 45, 17, 54, 23, 26, 28}:
            return k_mlp_0_0_output == 37
        elif q_mlp_0_0_output in {4, 36, 6, 41, 10, 43, 50, 57, 58}:
            return k_mlp_0_0_output == 49
        elif q_mlp_0_0_output in {9}:
            return k_mlp_0_0_output == 10
        elif q_mlp_0_0_output in {49, 13}:
            return k_mlp_0_0_output == 24
        elif q_mlp_0_0_output in {44, 14}:
            return k_mlp_0_0_output == 25
        elif q_mlp_0_0_output in {38, 15}:
            return k_mlp_0_0_output == 32
        elif q_mlp_0_0_output in {19}:
            return k_mlp_0_0_output == 28
        elif q_mlp_0_0_output in {24, 37, 55}:
            return k_mlp_0_0_output == 23
        elif q_mlp_0_0_output in {32, 25, 52, 39}:
            return k_mlp_0_0_output == 29
        elif q_mlp_0_0_output in {29}:
            return k_mlp_0_0_output == 43
        elif q_mlp_0_0_output in {30}:
            return k_mlp_0_0_output == 54
        elif q_mlp_0_0_output in {47}:
            return k_mlp_0_0_output == 44

    num_attn_1_4_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_2_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_0_output, mlp_0_1_output):
        if attn_0_0_output in {")", "<s>", "("}:
            return mlp_0_1_output == 52

    num_attn_1_5_pattern = select(mlp_0_1_outputs, attn_0_0_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_4_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_6_output, token):
        if attn_0_6_output in {")", "<s>", "("}:
            return token == ""

    num_attn_1_6_pattern = select(tokens, attn_0_6_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_1_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"("}:
            return mlp_0_0_output == 24
        elif attn_0_1_output in {")"}:
            return mlp_0_0_output == 37
        elif attn_0_1_output in {"<s>"}:
            return mlp_0_0_output == 32

    num_attn_1_7_pattern = select(mlp_0_0_outputs, attn_0_1_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_2_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_1_3_output):
        key = (attn_1_1_output, attn_1_3_output)
        if key in {(")", "<s>")}:
            return 59
        return 34

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_4_output, attn_0_6_output):
        key = (attn_1_4_output, attn_0_6_output)
        if key in {("<s>", "("), ("<s>", ")"), ("<s>", "<s>")}:
            return 2
        elif key in {("(", "(")}:
            return 1
        return 39

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_4_outputs, attn_0_6_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output):
        key = num_attn_1_4_output
        return 32

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_4_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_2_output, num_attn_1_6_output):
        key = (num_attn_0_2_output, num_attn_1_6_output)
        return 6

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_1_output, position):
        if attn_0_1_output in {"("}:
            return position == 3
        elif attn_0_1_output in {")"}:
            return position == 9
        elif attn_0_1_output in {"<s>"}:
            return position == 6

    attn_2_0_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 3
        elif attn_0_0_output in {")"}:
            return position == 10
        elif attn_0_0_output in {"<s>"}:
            return position == 2

    attn_2_1_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_4_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_6_output, mlp_0_1_output):
        if attn_0_6_output in {"("}:
            return mlp_0_1_output == 3
        elif attn_0_6_output in {")", "<s>"}:
            return mlp_0_1_output == 2

    attn_2_2_pattern = select_closest(mlp_0_1_outputs, attn_0_6_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_0_output, mlp_0_1_output):
        if attn_0_0_output in {"("}:
            return mlp_0_1_output == 46
        elif attn_0_0_output in {")", "<s>"}:
            return mlp_0_1_output == 2

    attn_2_3_pattern = select_closest(mlp_0_1_outputs, attn_0_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_6_output, mlp_0_1_output):
        if attn_0_6_output in {")", "("}:
            return mlp_0_1_output == 2
        elif attn_0_6_output in {"<s>"}:
            return mlp_0_1_output == 5

    attn_2_4_pattern = select_closest(mlp_0_1_outputs, attn_0_6_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 4
        elif attn_0_0_output in {")"}:
            return position == 6
        elif attn_0_0_output in {"<s>"}:
            return position == 3

    attn_2_5_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_2_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_0_output, mlp_0_1_output):
        if attn_0_0_output in {"("}:
            return mlp_0_1_output == 41
        elif attn_0_0_output in {")"}:
            return mlp_0_1_output == 2
        elif attn_0_0_output in {"<s>"}:
            return mlp_0_1_output == 9

    attn_2_6_pattern = select_closest(mlp_0_1_outputs, attn_0_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_5_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(token, position):
        if token in {"("}:
            return position == 8
        elif token in {")"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 1

    attn_2_7_pattern = select_closest(positions, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_3_output, attn_0_5_output):
        if attn_1_3_output in {")", "("}:
            return attn_0_5_output == ""
        elif attn_1_3_output in {"<s>"}:
            return attn_0_5_output == "<s>"

    num_attn_2_0_pattern = select(attn_0_5_outputs, attn_1_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_0_1_output, mlp_0_1_output):
        if num_mlp_0_1_output in {0}:
            return mlp_0_1_output == 0
        elif num_mlp_0_1_output in {1}:
            return mlp_0_1_output == 32
        elif num_mlp_0_1_output in {2}:
            return mlp_0_1_output == 9
        elif num_mlp_0_1_output in {3, 29}:
            return mlp_0_1_output == 27
        elif num_mlp_0_1_output in {4, 6, 8, 12, 46, 15, 17, 55, 58, 28}:
            return mlp_0_1_output == 37
        elif num_mlp_0_1_output in {32, 47, 5, 31}:
            return mlp_0_1_output == 35
        elif num_mlp_0_1_output in {37, 7, 10, 13, 45, 22}:
            return mlp_0_1_output == 1
        elif num_mlp_0_1_output in {9, 11}:
            return mlp_0_1_output == 57
        elif num_mlp_0_1_output in {14}:
            return mlp_0_1_output == 48
        elif num_mlp_0_1_output in {16, 26, 30, 39}:
            return mlp_0_1_output == 49
        elif num_mlp_0_1_output in {33, 18, 35, 34}:
            return mlp_0_1_output == 23
        elif num_mlp_0_1_output in {19}:
            return mlp_0_1_output == 6
        elif num_mlp_0_1_output in {20}:
            return mlp_0_1_output == 26
        elif num_mlp_0_1_output in {21}:
            return mlp_0_1_output == 51
        elif num_mlp_0_1_output in {56, 23}:
            return mlp_0_1_output == 53
        elif num_mlp_0_1_output in {24}:
            return mlp_0_1_output == 28
        elif num_mlp_0_1_output in {25}:
            return mlp_0_1_output == 14
        elif num_mlp_0_1_output in {27}:
            return mlp_0_1_output == 5
        elif num_mlp_0_1_output in {41, 36}:
            return mlp_0_1_output == 36
        elif num_mlp_0_1_output in {38}:
            return mlp_0_1_output == 8
        elif num_mlp_0_1_output in {40}:
            return mlp_0_1_output == 45
        elif num_mlp_0_1_output in {42}:
            return mlp_0_1_output == 33
        elif num_mlp_0_1_output in {50, 43}:
            return mlp_0_1_output == 38
        elif num_mlp_0_1_output in {44}:
            return mlp_0_1_output == 4
        elif num_mlp_0_1_output in {48}:
            return mlp_0_1_output == 7
        elif num_mlp_0_1_output in {49}:
            return mlp_0_1_output == 54
        elif num_mlp_0_1_output in {51}:
            return mlp_0_1_output == 25
        elif num_mlp_0_1_output in {52}:
            return mlp_0_1_output == 20
        elif num_mlp_0_1_output in {53}:
            return mlp_0_1_output == 22
        elif num_mlp_0_1_output in {54}:
            return mlp_0_1_output == 39
        elif num_mlp_0_1_output in {57}:
            return mlp_0_1_output == 42
        elif num_mlp_0_1_output in {59}:
            return mlp_0_1_output == 41

    num_attn_2_1_pattern = select(
        mlp_0_1_outputs, num_mlp_0_1_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_1_output, mlp_1_1_output):
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
            12,
            13,
            14,
            15,
            16,
            17,
            18,
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
            32,
            33,
            34,
            35,
            36,
            37,
            39,
            40,
            42,
            43,
            44,
            45,
            47,
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
            return mlp_1_1_output == 2
        elif mlp_0_1_output in {19, 50, 11, 54}:
            return mlp_1_1_output == 39
        elif mlp_0_1_output in {28, 31}:
            return mlp_1_1_output == 51
        elif mlp_0_1_output in {41, 38}:
            return mlp_1_1_output == 33
        elif mlp_0_1_output in {46}:
            return mlp_1_1_output == 22
        elif mlp_0_1_output in {48}:
            return mlp_1_1_output == 47

    num_attn_2_2_pattern = select(mlp_1_1_outputs, mlp_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_6_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_6_output, attn_0_5_output):
        if attn_1_6_output in {")", "<s>", "("}:
            return attn_0_5_output == ""

    num_attn_2_3_pattern = select(attn_0_5_outputs, attn_1_6_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_5_output, attn_1_6_output):
        if attn_0_5_output in {")", "<s>", "("}:
            return attn_1_6_output == ""

    num_attn_2_4_pattern = select(attn_1_6_outputs, attn_0_5_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(num_mlp_0_0_output, attn_1_7_output):
        if num_mlp_0_0_output in {
            0,
            3,
            5,
            8,
            9,
            12,
            13,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
            24,
            25,
            28,
            29,
            30,
            33,
            34,
            35,
            37,
            38,
            39,
            40,
            51,
            53,
            58,
        }:
            return attn_1_7_output == ")"
        elif num_mlp_0_0_output in {1, 4, 36, 41, 42, 18, 52, 57}:
            return attn_1_7_output == "<s>"
        elif num_mlp_0_0_output in {
            2,
            6,
            7,
            10,
            11,
            14,
            23,
            26,
            27,
            31,
            32,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            54,
            55,
            56,
            59,
        }:
            return attn_1_7_output == ""

    num_attn_2_5_pattern = select(
        attn_1_7_outputs, num_mlp_0_0_outputs, num_predicate_2_5
    )
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_7_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(num_mlp_0_0_output, num_mlp_1_0_output):
        if num_mlp_0_0_output in {0, 32, 35, 20}:
            return num_mlp_1_0_output == 22
        elif num_mlp_0_0_output in {1}:
            return num_mlp_1_0_output == 48
        elif num_mlp_0_0_output in {2, 50, 30}:
            return num_mlp_1_0_output == 14
        elif num_mlp_0_0_output in {3}:
            return num_mlp_1_0_output == 9
        elif num_mlp_0_0_output in {4, 36, 39}:
            return num_mlp_1_0_output == 16
        elif num_mlp_0_0_output in {5, 38}:
            return num_mlp_1_0_output == 25
        elif num_mlp_0_0_output in {37, 6}:
            return num_mlp_1_0_output == 53
        elif num_mlp_0_0_output in {15, 47, 7}:
            return num_mlp_1_0_output == 40
        elif num_mlp_0_0_output in {8, 55}:
            return num_mlp_1_0_output == 11
        elif num_mlp_0_0_output in {9}:
            return num_mlp_1_0_output == 35
        elif num_mlp_0_0_output in {10, 45}:
            return num_mlp_1_0_output == 12
        elif num_mlp_0_0_output in {43, 11}:
            return num_mlp_1_0_output == 41
        elif num_mlp_0_0_output in {12}:
            return num_mlp_1_0_output == 54
        elif num_mlp_0_0_output in {13}:
            return num_mlp_1_0_output == 36
        elif num_mlp_0_0_output in {40, 14}:
            return num_mlp_1_0_output == 13
        elif num_mlp_0_0_output in {16}:
            return num_mlp_1_0_output == 21
        elif num_mlp_0_0_output in {17}:
            return num_mlp_1_0_output == 6
        elif num_mlp_0_0_output in {18, 27, 53}:
            return num_mlp_1_0_output == 1
        elif num_mlp_0_0_output in {19, 29}:
            return num_mlp_1_0_output == 38
        elif num_mlp_0_0_output in {21}:
            return num_mlp_1_0_output == 26
        elif num_mlp_0_0_output in {22}:
            return num_mlp_1_0_output == 33
        elif num_mlp_0_0_output in {33, 23}:
            return num_mlp_1_0_output == 37
        elif num_mlp_0_0_output in {24, 48, 51}:
            return num_mlp_1_0_output == 39
        elif num_mlp_0_0_output in {25}:
            return num_mlp_1_0_output == 31
        elif num_mlp_0_0_output in {26}:
            return num_mlp_1_0_output == 58
        elif num_mlp_0_0_output in {28}:
            return num_mlp_1_0_output == 46
        elif num_mlp_0_0_output in {34, 31}:
            return num_mlp_1_0_output == 57
        elif num_mlp_0_0_output in {41}:
            return num_mlp_1_0_output == 51
        elif num_mlp_0_0_output in {42}:
            return num_mlp_1_0_output == 19
        elif num_mlp_0_0_output in {44}:
            return num_mlp_1_0_output == 45
        elif num_mlp_0_0_output in {46}:
            return num_mlp_1_0_output == 24
        elif num_mlp_0_0_output in {49}:
            return num_mlp_1_0_output == 47
        elif num_mlp_0_0_output in {52}:
            return num_mlp_1_0_output == 20
        elif num_mlp_0_0_output in {54}:
            return num_mlp_1_0_output == 23
        elif num_mlp_0_0_output in {56}:
            return num_mlp_1_0_output == 18
        elif num_mlp_0_0_output in {57}:
            return num_mlp_1_0_output == 3
        elif num_mlp_0_0_output in {58}:
            return num_mlp_1_0_output == 17
        elif num_mlp_0_0_output in {59}:
            return num_mlp_1_0_output == 10

    num_attn_2_6_pattern = select(
        num_mlp_1_0_outputs, num_mlp_0_0_outputs, num_predicate_2_6
    )
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(mlp_1_0_output, attn_0_5_output):
        if mlp_1_0_output in {0, 53}:
            return attn_0_5_output == "<s>"
        elif mlp_1_0_output in {
            1,
            2,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            27,
            29,
            31,
            32,
            33,
            36,
            37,
            38,
            41,
            42,
            43,
            44,
            46,
            47,
            50,
            55,
            56,
        }:
            return attn_0_5_output == ""
        elif mlp_1_0_output in {
            3,
            4,
            5,
            6,
            11,
            16,
            25,
            26,
            28,
            30,
            34,
            35,
            39,
            40,
            45,
            48,
            49,
            51,
            52,
            54,
            57,
            58,
            59,
        }:
            return attn_0_5_output == ")"

    num_attn_2_7_pattern = select(attn_0_5_outputs, mlp_1_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, attn_2_5_output):
        key = (attn_2_2_output, attn_2_5_output)
        if key in {(")", ")")}:
            return 11
        return 9

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_5_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_6_output, attn_2_1_output):
        key = (attn_2_6_output, attn_2_1_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")")}:
            return 50
        elif key in {("(", ")"), (")", "(")}:
            return 39
        return 9

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_6_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_1_5_output):
        key = (num_attn_1_2_output, num_attn_1_5_output)
        return 17

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_7_output, num_attn_2_5_output):
        key = (num_attn_2_7_output, num_attn_2_5_output)
        return 45

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_2_5_outputs)
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
        ]
    )
)
