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
        "output/length/rasp/reverse/trainlength20/s1/reverse_weights.csv",
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
        if q_position in {0, 5}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 15
        elif q_position in {2}:
            return k_position == 16
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 13
        elif q_position in {9, 6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {8, 10}:
            return k_position == 19
        elif q_position in {11}:
            return k_position == 5
        elif q_position in {12}:
            return k_position == 7
        elif q_position in {17, 13, 14}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 2
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 9
        elif q_position in {27, 20}:
            return k_position == 29
        elif q_position in {29, 21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 14
        elif q_position in {24}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {26}:
            return k_position == 22
        elif q_position in {28}:
            return k_position == 23

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1, 11}:
            return k_position == 7
        elif q_position in {2, 4}:
            return k_position == 8
        elif q_position in {3}:
            return k_position == 9
        elif q_position in {5}:
            return k_position == 12
        elif q_position in {6}:
            return k_position == 13
        elif q_position in {8, 7}:
            return k_position == 11
        elif q_position in {9, 10, 12}:
            return k_position == 4
        elif q_position in {16, 13, 14, 15}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 2
        elif q_position in {18, 19}:
            return k_position == 1
        elif q_position in {20}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {27, 29, 22}:
            return k_position == 20
        elif q_position in {25, 26, 23}:
            return k_position == 22
        elif q_position in {24, 28}:
            return k_position == 23

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1, 12}:
            return k_position == 18
        elif q_position in {2}:
            return k_position == 14
        elif q_position in {3}:
            return k_position == 16
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5, 6, 7, 9, 10, 11, 15, 16}:
            return k_position == 3
        elif q_position in {8, 13}:
            return k_position == 4
        elif q_position in {17, 14}:
            return k_position == 2
        elif q_position in {18, 19}:
            return k_position == 1
        elif q_position in {20, 23}:
            return k_position == 25
        elif q_position in {27, 21}:
            return k_position == 21
        elif q_position in {22}:
            return k_position == 29
        elif q_position in {24}:
            return k_position == 28
        elif q_position in {25, 28}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 24
        elif q_position in {29}:
            return k_position == 27

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 16
        elif q_position in {2}:
            return k_position == 15
        elif q_position in {3}:
            return k_position == 14
        elif q_position in {4, 12}:
            return k_position == 3
        elif q_position in {8, 5}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {24, 9}:
            return k_position == 19
        elif q_position in {10, 11, 13, 15, 16}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 4
        elif q_position in {17, 18, 19}:
            return k_position == 1
        elif q_position in {20, 28}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 27
        elif q_position in {22}:
            return k_position == 20
        elif q_position in {26, 29, 23}:
            return k_position == 29
        elif q_position in {25}:
            return k_position == 23
        elif q_position in {27}:
            return k_position == 26

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 13}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 17
        elif q_position in {2, 7}:
            return k_position == 10
        elif q_position in {3}:
            return k_position == 15
        elif q_position in {4}:
            return k_position == 14
        elif q_position in {5}:
            return k_position == 13
        elif q_position in {6, 8, 15, 16, 17}:
            return k_position == 2
        elif q_position in {9}:
            return k_position == 19
        elif q_position in {10, 11, 12, 14, 18, 19}:
            return k_position == 1
        elif q_position in {20}:
            return k_position == 22
        elif q_position in {21, 23}:
            return k_position == 26
        elif q_position in {22}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 29
        elif q_position in {25, 28}:
            return k_position == 28
        elif q_position in {26, 29}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 21

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 16, 13, 17}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 17
        elif q_position in {2, 18, 29}:
            return k_position == 18
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 11
        elif q_position in {5, 7, 8, 10, 20}:
            return k_position == 19
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {9, 12, 14}:
            return k_position == 2
        elif q_position in {11}:
            return k_position == 0
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 9
        elif q_position in {24, 27, 28, 21}:
            return k_position == 5
        elif q_position in {22}:
            return k_position == 27
        elif q_position in {23}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 25

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {3, 6}:
            return k_position == 13
        elif q_position in {4}:
            return k_position == 15
        elif q_position in {10, 5}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 19
        elif q_position in {8}:
            return k_position == 0
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {11, 15}:
            return k_position == 4
        elif q_position in {17, 13, 14}:
            return k_position == 2
        elif q_position in {16}:
            return k_position == 3
        elif q_position in {18, 19}:
            return k_position == 1
        elif q_position in {27, 20}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {24, 22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 29
        elif q_position in {26, 29}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 21

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 14}:
            return k_position == 5
        elif q_position in {1, 11}:
            return k_position == 8
        elif q_position in {2, 3, 13}:
            return k_position == 6
        elif q_position in {4, 15, 7}:
            return k_position == 4
        elif q_position in {5, 6}:
            return k_position == 7
        elif q_position in {8, 16}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 9
        elif q_position in {10, 12}:
            return k_position == 2
        elif q_position in {17, 18}:
            return k_position == 1
        elif q_position in {19}:
            return k_position == 13
        elif q_position in {20}:
            return k_position == 14
        elif q_position in {27, 21, 23}:
            return k_position == 20
        elif q_position in {28, 29, 22}:
            return k_position == 28
        elif q_position in {24}:
            return k_position == 24
        elif q_position in {25}:
            return k_position == 22
        elif q_position in {26}:
            return k_position == 26

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"0"}:
            return position == 20
        elif token in {"1"}:
            return position == 22
        elif token in {"2"}:
            return position == 25
        elif token in {"3"}:
            return position == 10
        elif token in {"4"}:
            return position == 27
        elif token in {"</s>"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 9

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 10
        elif q_position in {1, 3}:
            return k_position == 12
        elif q_position in {2, 4, 21}:
            return k_position == 29
        elif q_position in {27, 5}:
            return k_position == 24
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {16, 9, 28, 29}:
            return k_position == 20
        elif q_position in {10}:
            return k_position == 6
        elif q_position in {11}:
            return k_position == 28
        elif q_position in {12}:
            return k_position == 7
        elif q_position in {19, 13, 14}:
            return k_position == 27
        elif q_position in {20, 23, 15}:
            return k_position == 25
        elif q_position in {17}:
            return k_position == 21
        elif q_position in {18}:
            return k_position == 18
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {24, 26}:
            return k_position == 22
        elif q_position in {25}:
            return k_position == 26

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"0", "4"}:
            return position == 24
        elif token in {"1"}:
            return position == 20
        elif token in {"2"}:
            return position == 6
        elif token in {"3", "</s>"}:
            return position == 28
        elif token in {"<s>"}:
            return position == 26

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"0"}:
            return position == 22
        elif token in {"1"}:
            return position == 8
        elif token in {"2"}:
            return position == 27
        elif token in {"3"}:
            return position == 23
        elif token in {"4"}:
            return position == 20
        elif token in {"<s>", "</s>"}:
            return position == 9

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0, 26, 6, 22}:
            return k_position == 9
        elif q_position in {1, 21, 23}:
            return k_position == 23
        elif q_position in {2, 3, 11, 15, 19}:
            return k_position == 24
        elif q_position in {4, 28}:
            return k_position == 27
        elif q_position in {5}:
            return k_position == 22
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {8, 17, 10}:
            return k_position == 25
        elif q_position in {9, 18, 27}:
            return k_position == 21
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {16, 14}:
            return k_position == 7
        elif q_position in {24, 20, 29}:
            return k_position == 29
        elif q_position in {25}:
            return k_position == 8

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0, 1, 2, 6, 7, 8, 9, 25}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 28
        elif q_position in {27, 4}:
            return k_position == 23
        elif q_position in {5, 13, 14, 15, 17, 18}:
            return k_position == 19
        elif q_position in {16, 10, 21, 23}:
            return k_position == 20
        elif q_position in {11}:
            return k_position == 29
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {19}:
            return k_position == 27
        elif q_position in {20, 22}:
            return k_position == 22
        elif q_position in {24, 29}:
            return k_position == 5
        elif q_position in {26}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 25

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"0"}:
            return position == 10
        elif token in {"1"}:
            return position == 25
        elif token in {"2"}:
            return position == 24
        elif token in {"3", "</s>"}:
            return position == 8
        elif token in {"4"}:
            return position == 21
        elif token in {"<s>"}:
            return position == 9

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"0"}:
            return position == 8
        elif token in {"1"}:
            return position == 25
        elif token in {"2"}:
            return position == 22
        elif token in {"3"}:
            return position == 29
        elif token in {"4"}:
            return position == 27
        elif token in {"<s>", "</s>"}:
            return position == 9

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_7_output, attn_0_1_output):
        key = (attn_0_7_output, attn_0_1_output)
        if key in {
            ("0", "</s>"),
            ("1", "</s>"),
            ("2", "</s>"),
            ("3", "</s>"),
            ("</s>", "</s>"),
            ("<s>", "</s>"),
        }:
            return 6
        return 5

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_7_output, attn_0_1_output):
        key = (attn_0_7_output, attn_0_1_output)
        if key in {
            ("0", "</s>"),
            ("2", "</s>"),
            ("3", "</s>"),
            ("4", "</s>"),
            ("</s>", "3"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "</s>"),
        }:
            return 29
        elif key in {
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "4"),
        }:
            return 4
        elif key in {("</s>", "2"), ("</s>", "4"), ("<s>", "3"), ("<s>", "<s>")}:
            return 6
        elif key in {
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "<s>"),
        }:
            return 24
        elif key in {("4", "0"), ("4", "1"), ("4", "2"), ("4", "3"), ("4", "4")}:
            return 22
        elif key in {("1", "</s>")}:
            return 9
        elif key in {("4", "<s>")}:
            return 20
        return 2

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_1_outputs)
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
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_7_output):
        key = (num_attn_0_3_output, num_attn_0_7_output)
        return 2

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "<s>"
        elif attn_0_4_output in {"<s>", "2", "1", "</s>"}:
            return token == "3"
        elif attn_0_4_output in {"3"}:
            return token == "1"
        elif attn_0_4_output in {"4"}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, attn_0_4_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "4"
        elif attn_0_4_output in {"3", "1", "4"}:
            return token == ""
        elif attn_0_4_output in {"<s>", "2", "</s>"}:
            return token == "0"

    attn_1_1_pattern = select_closest(tokens, attn_0_4_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 20, 23, 24, 25, 26, 27, 28, 29}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {10, 12, 13, 6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8, 11, 14}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 13
        elif q_position in {16, 17}:
            return k_position == 17
        elif q_position in {18}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 14
        elif q_position in {21}:
            return k_position == 27
        elif q_position in {22}:
            return k_position == 23

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_5_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, token):
        if attn_0_3_output in {"2", "1", "4", "3", "</s>", "<s>", "0"}:
            return token == "3"

    attn_1_3_pattern = select_closest(tokens, attn_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_2_output, token):
        if attn_0_2_output in {"0", "<s>"}:
            return token == "2"
        elif attn_0_2_output in {"1"}:
            return token == "<s>"
        elif attn_0_2_output in {"2", "3", "</s>", "4"}:
            return token == ""

    attn_1_4_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_6_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_4_output, token):
        if attn_0_4_output in {"0", "3"}:
            return token == ""
        elif attn_0_4_output in {"1"}:
            return token == "1"
        elif attn_0_4_output in {"<s>", "2", "</s>"}:
            return token == "2"
        elif attn_0_4_output in {"4"}:
            return token == "4"

    attn_1_5_pattern = select_closest(tokens, attn_0_4_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_4_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 0
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {6, 7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 15
        elif q_position in {11, 12}:
            return k_position == 16
        elif q_position in {13, 14}:
            return k_position == 17
        elif q_position in {16, 24, 15}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 1
        elif q_position in {19}:
            return k_position == 3
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {25, 21}:
            return k_position == 23
        elif q_position in {29, 22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 29
        elif q_position in {26}:
            return k_position == 21
        elif q_position in {27}:
            return k_position == 22
        elif q_position in {28}:
            return k_position == 24

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_2_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_6_output, token):
        if attn_0_6_output in {"0", "</s>", "4"}:
            return token == ""
        elif attn_0_6_output in {"<s>", "1"}:
            return token == "2"
        elif attn_0_6_output in {"2"}:
            return token == "3"
        elif attn_0_6_output in {"3"}:
            return token == "0"

    attn_1_7_pattern = select_closest(tokens, attn_0_6_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_4_output, attn_0_0_output):
        if attn_0_4_output in {"0"}:
            return attn_0_0_output == "0"
        elif attn_0_4_output in {"2", "1", "4", "3", "</s>", "<s>"}:
            return attn_0_0_output == ""

    num_attn_1_0_pattern = select(attn_0_0_outputs, attn_0_4_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_4_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_attn_0_4_output, k_attn_0_4_output):
        if q_attn_0_4_output in {"0", "3", "<s>", "4"}:
            return k_attn_0_4_output == ""
        elif q_attn_0_4_output in {"1", "</s>"}:
            return k_attn_0_4_output == "<pad>"
        elif q_attn_0_4_output in {"2"}:
            return k_attn_0_4_output == "2"

    num_attn_1_1_pattern = select(attn_0_4_outputs, attn_0_4_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_4_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_attn_0_6_output, k_attn_0_6_output):
        if q_attn_0_6_output in {"2", "1", "4", "3", "</s>", "<s>", "0"}:
            return k_attn_0_6_output == ""

    num_attn_1_2_pattern = select(attn_0_6_outputs, attn_0_6_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_3_output, attn_0_0_output):
        if attn_0_3_output in {"1", "4", "3", "</s>", "<s>", "0"}:
            return attn_0_0_output == ""
        elif attn_0_3_output in {"2"}:
            return attn_0_0_output == "2"

    num_attn_1_3_pattern = select(attn_0_0_outputs, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_5_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 14, 22}:
            return k_mlp_0_0_output == 15
        elif q_mlp_0_0_output in {1}:
            return k_mlp_0_0_output == 17
        elif q_mlp_0_0_output in {25, 2, 26, 6}:
            return k_mlp_0_0_output == 26
        elif q_mlp_0_0_output in {3}:
            return k_mlp_0_0_output == 23
        elif q_mlp_0_0_output in {17, 4, 7}:
            return k_mlp_0_0_output == 24
        elif q_mlp_0_0_output in {9, 12, 5}:
            return k_mlp_0_0_output == 22
        elif q_mlp_0_0_output in {8}:
            return k_mlp_0_0_output == 1
        elif q_mlp_0_0_output in {10}:
            return k_mlp_0_0_output == 27
        elif q_mlp_0_0_output in {11, 21}:
            return k_mlp_0_0_output == 28
        elif q_mlp_0_0_output in {13, 16, 23, 24, 27}:
            return k_mlp_0_0_output == 21
        elif q_mlp_0_0_output in {15}:
            return k_mlp_0_0_output == 20
        elif q_mlp_0_0_output in {18}:
            return k_mlp_0_0_output == 12
        elif q_mlp_0_0_output in {19, 28}:
            return k_mlp_0_0_output == 4
        elif q_mlp_0_0_output in {20}:
            return k_mlp_0_0_output == 29
        elif q_mlp_0_0_output in {29}:
            return k_mlp_0_0_output == 3

    num_attn_1_4_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_4_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {0, 25, 20, 23}:
            return mlp_0_0_output == 12
        elif num_mlp_0_1_output in {1, 18}:
            return mlp_0_0_output == 29
        elif num_mlp_0_1_output in {2}:
            return mlp_0_0_output == 0
        elif num_mlp_0_1_output in {3, 4}:
            return mlp_0_0_output == 21
        elif num_mlp_0_1_output in {21, 11, 5}:
            return mlp_0_0_output == 10
        elif num_mlp_0_1_output in {6}:
            return mlp_0_0_output == 17
        elif num_mlp_0_1_output in {16, 7}:
            return mlp_0_0_output == 9
        elif num_mlp_0_1_output in {8, 9, 28}:
            return mlp_0_0_output == 14
        elif num_mlp_0_1_output in {10, 14}:
            return mlp_0_0_output == 26
        elif num_mlp_0_1_output in {12}:
            return mlp_0_0_output == 1
        elif num_mlp_0_1_output in {13}:
            return mlp_0_0_output == 19
        elif num_mlp_0_1_output in {22, 15}:
            return mlp_0_0_output == 7
        elif num_mlp_0_1_output in {17}:
            return mlp_0_0_output == 27
        elif num_mlp_0_1_output in {19, 29}:
            return mlp_0_0_output == 13
        elif num_mlp_0_1_output in {24, 27}:
            return mlp_0_0_output == 11
        elif num_mlp_0_1_output in {26}:
            return mlp_0_0_output == 16

    num_attn_1_5_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_2_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_3_output, attn_0_4_output):
        if attn_0_3_output in {"2", "1", "4", "</s>", "<s>", "0"}:
            return attn_0_4_output == ""
        elif attn_0_3_output in {"3"}:
            return attn_0_4_output == "3"

    num_attn_1_6_pattern = select(attn_0_4_outputs, attn_0_3_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_5_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_3_output, attn_0_4_output):
        if attn_0_3_output in {"2", "1", "3", "</s>", "<s>", "0"}:
            return attn_0_4_output == ""
        elif attn_0_3_output in {"4"}:
            return attn_0_4_output == "4"

    num_attn_1_7_pattern = select(attn_0_4_outputs, attn_0_3_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_5_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_1_output, attn_0_7_output):
        key = (attn_0_1_output, attn_0_7_output)
        if key in {
            ("0", "0"),
            ("1", "0"),
            ("1", "<s>"),
            ("2", "0"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "<s>"),
            ("<s>", "0"),
            ("<s>", "<s>"),
        }:
            return 3
        elif key in {("1", "</s>"), ("2", "</s>"), ("<s>", "</s>")}:
            return 27
        elif key in {("</s>", "2")}:
            return 1
        elif key in {
            ("</s>", "1"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
        }:
            return 15
        elif key in {("<s>", "1")}:
            return 19
        elif key in {("</s>", "0")}:
            return 22
        return 18

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_7_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_3_output, attn_0_0_output):
        key = (attn_1_3_output, attn_0_0_output)
        return 23

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_5_output, num_attn_1_2_output):
        key = (num_attn_0_5_output, num_attn_1_2_output)
        if key in {
            (31, 58),
            (31, 59),
            (32, 56),
            (32, 57),
            (32, 58),
            (32, 59),
            (33, 54),
            (33, 55),
            (33, 56),
            (33, 57),
            (33, 58),
            (33, 59),
            (34, 52),
            (34, 53),
            (34, 54),
            (34, 55),
            (34, 56),
            (34, 57),
            (34, 58),
            (34, 59),
            (35, 50),
            (35, 51),
            (35, 52),
            (35, 53),
            (35, 54),
            (35, 55),
            (35, 56),
            (35, 57),
            (35, 58),
            (35, 59),
            (36, 48),
            (36, 49),
            (36, 50),
            (36, 51),
            (36, 52),
            (36, 53),
            (36, 54),
            (36, 55),
            (36, 56),
            (36, 57),
            (36, 58),
            (36, 59),
            (37, 46),
            (37, 47),
            (37, 48),
            (37, 49),
            (37, 50),
            (37, 51),
            (37, 52),
            (37, 53),
            (37, 54),
            (37, 55),
            (37, 56),
            (37, 57),
            (37, 58),
            (37, 59),
            (38, 45),
            (38, 46),
            (38, 47),
            (38, 48),
            (38, 49),
            (38, 50),
            (38, 51),
            (38, 52),
            (38, 53),
            (38, 54),
            (38, 55),
            (38, 56),
            (38, 57),
            (38, 58),
            (38, 59),
            (39, 43),
            (39, 44),
            (39, 45),
            (39, 46),
            (39, 47),
            (39, 48),
            (39, 49),
            (39, 50),
            (39, 51),
            (39, 52),
            (39, 53),
            (39, 54),
            (39, 55),
            (39, 56),
            (39, 57),
            (39, 58),
            (39, 59),
            (40, 41),
            (40, 42),
            (40, 43),
            (40, 44),
            (40, 45),
            (40, 46),
            (40, 47),
            (40, 48),
            (40, 49),
            (40, 50),
            (40, 51),
            (40, 52),
            (40, 53),
            (40, 54),
            (40, 55),
            (40, 56),
            (40, 57),
            (40, 58),
            (40, 59),
            (41, 39),
            (41, 40),
            (41, 41),
            (41, 42),
            (41, 43),
            (41, 44),
            (41, 45),
            (41, 46),
            (41, 47),
            (41, 48),
            (41, 49),
            (41, 50),
            (41, 51),
            (41, 52),
            (41, 53),
            (41, 54),
            (41, 55),
            (41, 56),
            (41, 57),
            (41, 58),
            (41, 59),
            (42, 37),
            (42, 38),
            (42, 39),
            (42, 40),
            (42, 41),
            (42, 42),
            (42, 43),
            (42, 44),
            (42, 45),
            (42, 46),
            (42, 47),
            (42, 48),
            (42, 49),
            (42, 50),
            (42, 51),
            (42, 52),
            (42, 53),
            (42, 54),
            (42, 55),
            (42, 56),
            (42, 57),
            (42, 58),
            (42, 59),
            (43, 35),
            (43, 36),
            (43, 37),
            (43, 38),
            (43, 39),
            (43, 40),
            (43, 41),
            (43, 42),
            (43, 43),
            (43, 44),
            (43, 45),
            (43, 46),
            (43, 47),
            (43, 48),
            (43, 49),
            (43, 50),
            (43, 51),
            (43, 52),
            (43, 53),
            (43, 54),
            (43, 55),
            (43, 56),
            (43, 57),
            (43, 58),
            (43, 59),
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
            (44, 44),
            (44, 45),
            (44, 46),
            (44, 47),
            (44, 48),
            (44, 49),
            (44, 50),
            (44, 51),
            (44, 52),
            (44, 53),
            (44, 54),
            (44, 55),
            (44, 56),
            (44, 57),
            (44, 58),
            (44, 59),
            (45, 31),
            (45, 32),
            (45, 33),
            (45, 34),
            (45, 35),
            (45, 36),
            (45, 37),
            (45, 38),
            (45, 39),
            (45, 40),
            (45, 41),
            (45, 42),
            (45, 43),
            (45, 44),
            (45, 45),
            (45, 46),
            (45, 47),
            (45, 48),
            (45, 49),
            (45, 50),
            (45, 51),
            (45, 52),
            (45, 53),
            (45, 54),
            (45, 55),
            (45, 56),
            (45, 57),
            (45, 58),
            (45, 59),
            (46, 29),
            (46, 30),
            (46, 31),
            (46, 32),
            (46, 33),
            (46, 34),
            (46, 35),
            (46, 36),
            (46, 37),
            (46, 38),
            (46, 39),
            (46, 40),
            (46, 41),
            (46, 42),
            (46, 43),
            (46, 44),
            (46, 45),
            (46, 46),
            (46, 47),
            (46, 48),
            (46, 49),
            (46, 50),
            (46, 51),
            (46, 52),
            (46, 53),
            (46, 54),
            (46, 55),
            (46, 56),
            (46, 57),
            (46, 58),
            (46, 59),
            (47, 27),
            (47, 28),
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
            (47, 50),
            (47, 51),
            (47, 52),
            (47, 53),
            (47, 54),
            (47, 55),
            (47, 56),
            (47, 57),
            (47, 58),
            (47, 59),
            (48, 26),
            (48, 27),
            (48, 28),
            (48, 29),
            (48, 30),
            (48, 31),
            (48, 32),
            (48, 33),
            (48, 34),
            (48, 35),
            (48, 36),
            (48, 37),
            (48, 38),
            (48, 39),
            (48, 40),
            (48, 41),
            (48, 42),
            (48, 43),
            (48, 44),
            (48, 45),
            (48, 46),
            (48, 47),
            (48, 48),
            (48, 49),
            (48, 50),
            (48, 51),
            (48, 52),
            (48, 53),
            (48, 54),
            (48, 55),
            (48, 56),
            (48, 57),
            (48, 58),
            (48, 59),
            (49, 24),
            (49, 25),
            (49, 26),
            (49, 27),
            (49, 28),
            (49, 29),
            (49, 30),
            (49, 31),
            (49, 32),
            (49, 33),
            (49, 34),
            (49, 35),
            (49, 36),
            (49, 37),
            (49, 38),
            (49, 39),
            (49, 40),
            (49, 41),
            (49, 42),
            (49, 43),
            (49, 44),
            (49, 45),
            (49, 46),
            (49, 47),
            (49, 48),
            (49, 49),
            (49, 50),
            (49, 51),
            (49, 52),
            (49, 53),
            (49, 54),
            (49, 55),
            (49, 56),
            (49, 57),
            (49, 58),
            (49, 59),
            (50, 22),
            (50, 23),
            (50, 24),
            (50, 25),
            (50, 26),
            (50, 27),
            (50, 28),
            (50, 29),
            (50, 30),
            (50, 31),
            (50, 32),
            (50, 33),
            (50, 34),
            (50, 35),
            (50, 36),
            (50, 37),
            (50, 38),
            (50, 39),
            (50, 40),
            (50, 41),
            (50, 42),
            (50, 43),
            (50, 44),
            (50, 45),
            (50, 46),
            (50, 47),
            (50, 48),
            (50, 49),
            (50, 50),
            (50, 51),
            (50, 52),
            (50, 53),
            (50, 54),
            (50, 55),
            (50, 56),
            (50, 57),
            (50, 58),
            (50, 59),
            (51, 20),
            (51, 21),
            (51, 22),
            (51, 23),
            (51, 24),
            (51, 25),
            (51, 26),
            (51, 27),
            (51, 28),
            (51, 29),
            (51, 30),
            (51, 31),
            (51, 32),
            (51, 33),
            (51, 34),
            (51, 35),
            (51, 36),
            (51, 37),
            (51, 38),
            (51, 39),
            (51, 40),
            (51, 41),
            (51, 42),
            (51, 43),
            (51, 44),
            (51, 45),
            (51, 46),
            (51, 47),
            (51, 48),
            (51, 49),
            (51, 50),
            (51, 51),
            (51, 52),
            (51, 53),
            (51, 54),
            (51, 55),
            (51, 56),
            (51, 57),
            (51, 58),
            (51, 59),
            (52, 18),
            (52, 19),
            (52, 20),
            (52, 21),
            (52, 22),
            (52, 23),
            (52, 24),
            (52, 25),
            (52, 26),
            (52, 27),
            (52, 28),
            (52, 29),
            (52, 30),
            (52, 31),
            (52, 32),
            (52, 33),
            (52, 34),
            (52, 35),
            (52, 36),
            (52, 37),
            (52, 38),
            (52, 39),
            (52, 40),
            (52, 41),
            (52, 42),
            (52, 43),
            (52, 44),
            (52, 45),
            (52, 46),
            (52, 47),
            (52, 48),
            (52, 49),
            (52, 50),
            (52, 51),
            (52, 52),
            (52, 53),
            (52, 54),
            (52, 55),
            (52, 56),
            (52, 57),
            (52, 58),
            (52, 59),
            (53, 16),
            (53, 17),
            (53, 18),
            (53, 19),
            (53, 20),
            (53, 21),
            (53, 22),
            (53, 23),
            (53, 24),
            (53, 25),
            (53, 26),
            (53, 27),
            (53, 28),
            (53, 29),
            (53, 30),
            (53, 31),
            (53, 32),
            (53, 33),
            (53, 34),
            (53, 35),
            (53, 36),
            (53, 37),
            (53, 38),
            (53, 39),
            (53, 40),
            (53, 41),
            (53, 42),
            (53, 43),
            (53, 44),
            (53, 45),
            (53, 46),
            (53, 47),
            (53, 48),
            (53, 49),
            (53, 50),
            (53, 51),
            (53, 52),
            (53, 53),
            (53, 54),
            (53, 55),
            (53, 56),
            (53, 57),
            (53, 58),
            (53, 59),
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
            (54, 25),
            (54, 26),
            (54, 27),
            (54, 28),
            (54, 29),
            (54, 30),
            (54, 31),
            (54, 32),
            (54, 33),
            (54, 34),
            (54, 35),
            (54, 36),
            (54, 37),
            (54, 38),
            (54, 39),
            (54, 40),
            (54, 41),
            (54, 42),
            (54, 43),
            (54, 44),
            (54, 45),
            (54, 46),
            (54, 47),
            (54, 48),
            (54, 49),
            (54, 50),
            (54, 51),
            (54, 52),
            (54, 53),
            (54, 54),
            (54, 55),
            (54, 56),
            (54, 57),
            (54, 58),
            (54, 59),
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
            (55, 25),
            (55, 26),
            (55, 27),
            (55, 28),
            (55, 29),
            (55, 30),
            (55, 31),
            (55, 32),
            (55, 33),
            (55, 34),
            (55, 35),
            (55, 36),
            (55, 37),
            (55, 38),
            (55, 39),
            (55, 40),
            (55, 41),
            (55, 42),
            (55, 43),
            (55, 44),
            (55, 45),
            (55, 46),
            (55, 47),
            (55, 48),
            (55, 49),
            (55, 50),
            (55, 51),
            (55, 52),
            (55, 53),
            (55, 54),
            (55, 55),
            (55, 56),
            (55, 57),
            (55, 58),
            (55, 59),
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
            (56, 26),
            (56, 27),
            (56, 28),
            (56, 29),
            (56, 30),
            (56, 31),
            (56, 32),
            (56, 33),
            (56, 34),
            (56, 35),
            (56, 36),
            (56, 37),
            (56, 38),
            (56, 39),
            (56, 40),
            (56, 41),
            (56, 42),
            (56, 43),
            (56, 44),
            (56, 45),
            (56, 46),
            (56, 47),
            (56, 48),
            (56, 49),
            (56, 50),
            (56, 51),
            (56, 52),
            (56, 53),
            (56, 54),
            (56, 55),
            (56, 56),
            (56, 57),
            (56, 58),
            (56, 59),
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
            (57, 26),
            (57, 27),
            (57, 28),
            (57, 29),
            (57, 30),
            (57, 31),
            (57, 32),
            (57, 33),
            (57, 34),
            (57, 35),
            (57, 36),
            (57, 37),
            (57, 38),
            (57, 39),
            (57, 40),
            (57, 41),
            (57, 42),
            (57, 43),
            (57, 44),
            (57, 45),
            (57, 46),
            (57, 47),
            (57, 48),
            (57, 49),
            (57, 50),
            (57, 51),
            (57, 52),
            (57, 53),
            (57, 54),
            (57, 55),
            (57, 56),
            (57, 57),
            (57, 58),
            (57, 59),
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
            (58, 27),
            (58, 28),
            (58, 29),
            (58, 30),
            (58, 31),
            (58, 32),
            (58, 33),
            (58, 34),
            (58, 35),
            (58, 36),
            (58, 37),
            (58, 38),
            (58, 39),
            (58, 40),
            (58, 41),
            (58, 42),
            (58, 43),
            (58, 44),
            (58, 45),
            (58, 46),
            (58, 47),
            (58, 48),
            (58, 49),
            (58, 50),
            (58, 51),
            (58, 52),
            (58, 53),
            (58, 54),
            (58, 55),
            (58, 56),
            (58, 57),
            (58, 58),
            (58, 59),
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
            (59, 27),
            (59, 28),
            (59, 29),
            (59, 30),
            (59, 31),
            (59, 32),
            (59, 33),
            (59, 34),
            (59, 35),
            (59, 36),
            (59, 37),
            (59, 38),
            (59, 39),
            (59, 40),
            (59, 41),
            (59, 42),
            (59, 43),
            (59, 44),
            (59, 45),
            (59, 46),
            (59, 47),
            (59, 48),
            (59, 49),
            (59, 50),
            (59, 51),
            (59, 52),
            (59, 53),
            (59, 54),
            (59, 55),
            (59, 56),
            (59, 57),
            (59, 58),
            (59, 59),
        }:
            return 7
        return 5

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_7_output):
        key = num_attn_1_7_output
        return 19

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_1_7_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_3_output, token):
        if attn_0_3_output in {"</s>", "0", "1", "4"}:
            return token == ""
        elif attn_0_3_output in {"2"}:
            return token == "0"
        elif attn_0_3_output in {"3"}:
            return token == "1"
        elif attn_0_3_output in {"<s>"}:
            return token == "3"

    attn_2_0_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_4_output, position):
        if attn_0_4_output in {"0"}:
            return position == 5
        elif attn_0_4_output in {"2", "1"}:
            return position == 19
        elif attn_0_4_output in {"3"}:
            return position == 18
        elif attn_0_4_output in {"</s>", "4"}:
            return position == 4
        elif attn_0_4_output in {"<s>"}:
            return position == 26

    attn_2_1_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_6_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 1, 5, 9, 10, 16, 17, 19, 20, 21, 22, 23, 25, 27, 28, 29}:
            return token == ""
        elif position in {2, 4, 7, 8, 11}:
            return token == "</s>"
        elif position in {24, 3}:
            return token == "<s>"
        elif position in {18, 6}:
            return token == "0"
        elif position in {12, 13, 15}:
            return token == "3"
        elif position in {14}:
            return token == "1"
        elif position in {26}:
            return token == "<pad>"

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_6_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_4_output, token):
        if attn_0_4_output in {"1", "4", "</s>", "<s>", "0"}:
            return token == "2"
        elif attn_0_4_output in {"2"}:
            return token == "4"
        elif attn_0_4_output in {"3"}:
            return token == ""

    attn_2_3_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, tokens)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(position, token):
        if position in {0, 12, 21, 22, 24, 26, 27, 28, 29}:
            return token == ""
        elif position in {1, 18}:
            return token == "1"
        elif position in {17, 2, 3}:
            return token == "4"
        elif position in {4, 5}:
            return token == "3"
        elif position in {6, 7, 8, 9, 10, 11}:
            return token == "</s>"
        elif position in {16, 13, 14}:
            return token == "<s>"
        elif position in {15, 19, 20, 23, 25}:
            return token == "2"

    attn_2_4_pattern = select_closest(tokens, positions, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_4_output, token):
        if attn_0_4_output in {"0", "3"}:
            return token == "<s>"
        elif attn_0_4_output in {"</s>", "1", "4"}:
            return token == ""
        elif attn_0_4_output in {"2"}:
            return token == "1"
        elif attn_0_4_output in {"<s>"}:
            return token == "</s>"

    attn_2_5_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_3_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"2", "0", "3", "1"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "1"
        elif q_token in {"<s>", "</s>"}:
            return k_token == ""

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, tokens)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_3_output, token):
        if attn_0_3_output in {"0", "<s>"}:
            return token == "0"
        elif attn_0_3_output in {"1"}:
            return token == "1"
        elif attn_0_3_output in {"2"}:
            return token == "2"
        elif attn_0_3_output in {"3"}:
            return token == "3"
        elif attn_0_3_output in {"4"}:
            return token == "4"
        elif attn_0_3_output in {"</s>"}:
            return token == "</s>"

    attn_2_7_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_2_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_0_0_output, attn_1_7_output):
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
        }:
            return attn_1_7_output == ""
        elif num_mlp_0_0_output in {10, 20, 22}:
            return attn_1_7_output == "<pad>"

    num_attn_2_0_pattern = select(
        attn_1_7_outputs, num_mlp_0_0_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_2_output, attn_0_7_output):
        if attn_1_2_output in {"1", "4", "</s>", "<s>", "0"}:
            return attn_0_7_output == ""
        elif attn_1_2_output in {"2"}:
            return attn_0_7_output == "<pad>"
        elif attn_1_2_output in {"3"}:
            return attn_0_7_output == "3"

    num_attn_2_1_pattern = select(attn_0_7_outputs, attn_1_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_2_output, attn_0_7_output):
        if attn_1_2_output in {"1", "4", "3", "</s>", "<s>", "0"}:
            return attn_0_7_output == ""
        elif attn_1_2_output in {"2"}:
            return attn_0_7_output == "2"

    num_attn_2_2_pattern = select(attn_0_7_outputs, attn_1_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_6_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_4_output, attn_1_2_output):
        if attn_0_4_output in {"2", "4", "3", "</s>", "<s>", "0"}:
            return attn_1_2_output == ""
        elif attn_0_4_output in {"1"}:
            return attn_1_2_output == "1"

    num_attn_2_3_pattern = select(attn_1_2_outputs, attn_0_4_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_4_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_3_output, attn_0_4_output):
        if attn_0_3_output in {"2", "3", "</s>", "<s>", "0"}:
            return attn_0_4_output == ""
        elif attn_0_3_output in {"1"}:
            return attn_0_4_output == "1"
        elif attn_0_3_output in {"4"}:
            return attn_0_4_output == "<pad>"

    num_attn_2_4_pattern = select(attn_0_4_outputs, attn_0_3_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_5_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_3_output, attn_1_2_output):
        if attn_0_3_output in {"0"}:
            return attn_1_2_output == "0"
        elif attn_0_3_output in {"2", "1", "4", "3", "</s>", "<s>"}:
            return attn_1_2_output == ""

    num_attn_2_5_pattern = select(attn_1_2_outputs, attn_0_3_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_5_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_4_output, attn_1_2_output):
        if attn_0_4_output in {"2", "1", "4", "</s>", "<s>", "0"}:
            return attn_1_2_output == ""
        elif attn_0_4_output in {"3"}:
            return attn_1_2_output == "3"

    num_attn_2_6_pattern = select(attn_1_2_outputs, attn_0_4_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_4_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_2_output, attn_0_7_output):
        if attn_1_2_output in {"0"}:
            return attn_0_7_output == "0"
        elif attn_1_2_output in {"<s>", "</s>", "1", "4"}:
            return attn_0_7_output == ""
        elif attn_1_2_output in {"2", "3"}:
            return attn_0_7_output == "<pad>"

    num_attn_2_7_pattern = select(attn_0_7_outputs, attn_1_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_1_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_4_output, attn_2_3_output):
        key = (attn_2_4_output, attn_2_3_output)
        return 1

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_4_outputs, attn_2_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(token, attn_1_4_output):
        key = (token, attn_1_4_output)
        return 0

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(tokens, attn_1_4_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_7_output, num_attn_2_7_output):
        key = (num_attn_1_7_output, num_attn_2_7_output)
        return 25

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_2_7_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_4_output, num_attn_1_2_output):
        key = (num_attn_2_4_output, num_attn_1_2_output)
        return 11

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_4_outputs, num_attn_1_2_outputs)
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


print(run(["<s>", "3", "4", "0", "1", "3", "0", "</s>"]))
