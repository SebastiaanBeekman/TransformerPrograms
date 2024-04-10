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
        "output/length/rasp/reverse/trainlength20/s2/reverse_weights.csv",
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
            return k_position == 13
        elif q_position in {1}:
            return k_position == 17
        elif q_position in {2}:
            return k_position == 16
        elif q_position in {3}:
            return k_position == 15
        elif q_position in {4}:
            return k_position == 14
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 0
        elif q_position in {9, 13, 14, 15}:
            return k_position == 3
        elif q_position in {10, 11}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 4
        elif q_position in {16, 18, 19}:
            return k_position == 1
        elif q_position in {17}:
            return k_position == 2
        elif q_position in {20}:
            return k_position == 29
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {28, 22}:
            return k_position == 21
        elif q_position in {24, 27, 23}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {29}:
            return k_position == 26

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {2}:
            return k_position == 11
        elif q_position in {3}:
            return k_position == 16
        elif q_position in {4, 6}:
            return k_position == 13
        elif q_position in {15, 5, 7}:
            return k_position == 19
        elif q_position in {8, 12}:
            return k_position == 3
        elif q_position in {9, 10, 11}:
            return k_position == 0
        elif q_position in {17, 18, 13}:
            return k_position == 1
        elif q_position in {14}:
            return k_position == 5
        elif q_position in {16, 19}:
            return k_position == 2
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {24, 21}:
            return k_position == 28
        elif q_position in {22}:
            return k_position == 22
        elif q_position in {23}:
            return k_position == 29
        elif q_position in {25, 27}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 27
        elif q_position in {28}:
            return k_position == 20
        elif q_position in {29}:
            return k_position == 25

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 19
        elif q_position in {1, 3}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {4}:
            return k_position == 11
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {6, 7, 9, 10, 12, 13, 14, 18, 19}:
            return k_position == 1
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {16, 11}:
            return k_position == 3
        elif q_position in {15}:
            return k_position == 4
        elif q_position in {17}:
            return k_position == 2
        elif q_position in {24, 25, 20}:
            return k_position == 23
        elif q_position in {21}:
            return k_position == 29
        elif q_position in {26, 29, 22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 27
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 20

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 14
        elif q_position in {1, 2, 4}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 13
        elif q_position in {6, 8, 12, 13, 14}:
            return k_position == 19
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {19, 10, 18}:
            return k_position == 1
        elif q_position in {16, 11}:
            return k_position == 3
        elif q_position in {15}:
            return k_position == 4
        elif q_position in {17}:
            return k_position == 2
        elif q_position in {20, 23}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 29
        elif q_position in {22}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28, 29}:
            return k_position == 20

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 18
        elif q_position in {2}:
            return k_position == 15
        elif q_position in {3}:
            return k_position == 11
        elif q_position in {4}:
            return k_position == 12
        elif q_position in {5, 14}:
            return k_position == 3
        elif q_position in {11, 6}:
            return k_position == 0
        elif q_position in {7}:
            return k_position == 19
        elif q_position in {8, 9, 12}:
            return k_position == 7
        elif q_position in {10, 13}:
            return k_position == 6
        elif q_position in {17, 18, 15}:
            return k_position == 1
        elif q_position in {16}:
            return k_position == 2
        elif q_position in {19}:
            return k_position == 13
        elif q_position in {20, 28}:
            return k_position == 25
        elif q_position in {21}:
            return k_position == 29
        elif q_position in {25, 22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 22
        elif q_position in {27}:
            return k_position == 28
        elif q_position in {29}:
            return k_position == 26

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 19
        elif q_position in {1}:
            return k_position == 12
        elif q_position in {2}:
            return k_position == 13
        elif q_position in {3}:
            return k_position == 10
        elif q_position in {5, 6, 7, 8, 19}:
            return k_position == 0
        elif q_position in {9, 11}:
            return k_position == 2
        elif q_position in {10}:
            return k_position == 3
        elif q_position in {12, 14, 15, 16, 17, 18}:
            return k_position == 1
        elif q_position in {13}:
            return k_position == 4
        elif q_position in {20}:
            return k_position == 20
        elif q_position in {25, 27, 21}:
            return k_position == 29
        elif q_position in {28, 22}:
            return k_position == 26
        elif q_position in {29, 23}:
            return k_position == 25
        elif q_position in {24, 26}:
            return k_position == 21

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "3"
        elif q_token in {"3", "1", "4", "2"}:
            return k_token == "0"
        elif q_token in {"</s>"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "4"

    attn_0_6_pattern = select_closest(tokens, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 5, 6, 7, 9}:
            return k_position == 19
        elif q_position in {1, 20}:
            return k_position == 18
        elif q_position in {2}:
            return k_position == 17
        elif q_position in {3, 4}:
            return k_position == 15
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {10, 12, 13, 14, 15, 17}:
            return k_position == 2
        elif q_position in {19, 18, 11}:
            return k_position == 1
        elif q_position in {16}:
            return k_position == 3
        elif q_position in {28, 21, 22}:
            return k_position == 25
        elif q_position in {23}:
            return k_position == 28
        elif q_position in {24}:
            return k_position == 21
        elif q_position in {25, 26}:
            return k_position == 22
        elif q_position in {27}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 20

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 20
        elif q_position in {2}:
            return k_position == 11
        elif q_position in {3, 7}:
            return k_position == 9
        elif q_position in {4, 6}:
            return k_position == 10
        elif q_position in {19, 5}:
            return k_position == 14
        elif q_position in {8, 29}:
            return k_position == 23
        elif q_position in {24, 9, 20}:
            return k_position == 21
        elif q_position in {10}:
            return k_position == 29
        elif q_position in {18, 11, 15}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 25
        elif q_position in {21, 13}:
            return k_position == 26
        elif q_position in {14}:
            return k_position == 8
        elif q_position in {16}:
            return k_position == 5
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {22, 23}:
            return k_position == 24
        elif q_position in {25, 26, 28}:
            return k_position == 28
        elif q_position in {27}:
            return k_position == 22

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {1, 2}:
            return k_position == 29
        elif q_position in {3, 22}:
            return k_position == 25
        elif q_position in {24, 19, 4, 15}:
            return k_position == 22
        elif q_position in {25, 5}:
            return k_position == 26
        elif q_position in {6, 7, 8, 17, 20}:
            return k_position == 24
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {10, 13, 14}:
            return k_position == 16
        elif q_position in {11}:
            return k_position == 23
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {16}:
            return k_position == 2
        elif q_position in {29, 18, 27, 21}:
            return k_position == 27
        elif q_position in {23}:
            return k_position == 21
        elif q_position in {26}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 20

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"</s>", "<s>", "3", "0"}:
            return position == 8
        elif token in {"1"}:
            return position == 22
        elif token in {"2"}:
            return position == 25
        elif token in {"4"}:
            return position == 21

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 1, 2, 7, 22, 27}:
            return k_position == 8
        elif q_position in {3}:
            return k_position == 11
        elif q_position in {25, 19, 4}:
            return k_position == 27
        elif q_position in {17, 5}:
            return k_position == 28
        elif q_position in {24, 18, 29, 6}:
            return k_position == 20
        elif q_position in {8, 16, 21}:
            return k_position == 23
        elif q_position in {9, 13}:
            return k_position == 17
        elif q_position in {10, 11}:
            return k_position == 12
        elif q_position in {12, 14}:
            return k_position == 16
        elif q_position in {28, 15}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 29
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {26}:
            return k_position == 0

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"0"}:
            return position == 21
        elif token in {"<s>", "1"}:
            return position == 8
        elif token in {"2"}:
            return position == 29
        elif token in {"3"}:
            return position == 23
        elif token in {"4"}:
            return position == 22
        elif token in {"</s>"}:
            return position == 20

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"0"}:
            return position == 23
        elif token in {"1"}:
            return position == 29
        elif token in {"2"}:
            return position == 28
        elif token in {"<s>", "3", "</s>"}:
            return position == 8
        elif token in {"4"}:
            return position == 22

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"0"}:
            return position == 20
        elif token in {"1"}:
            return position == 22
        elif token in {"2"}:
            return position == 28
        elif token in {"3"}:
            return position == 21
        elif token in {"</s>", "<s>", "4"}:
            return position == 8

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 9
        elif q_position in {2}:
            return k_position == 10
        elif q_position in {27, 25, 3, 5}:
            return k_position == 23
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {24, 15, 7}:
            return k_position == 26
        elif q_position in {8, 18, 29}:
            return k_position == 19
        elif q_position in {9}:
            return k_position == 18
        elif q_position in {10, 11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 29
        elif q_position in {13}:
            return k_position == 28
        elif q_position in {17, 26, 28, 14}:
            return k_position == 27
        elif q_position in {16, 19, 23}:
            return k_position == 21
        elif q_position in {20, 22}:
            return k_position == 22
        elif q_position in {21}:
            return k_position == 25

    num_attn_0_7_pattern = select(positions, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_7_output, attn_0_6_output):
        key = (attn_0_7_output, attn_0_6_output)
        return 6

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_6_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_3_output):
        key = (attn_0_1_output, attn_0_3_output)
        if key in {
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("2", "<s>"),
            ("3", "3"),
            ("4", "3"),
            ("</s>", "2"),
            ("<s>", "3"),
        }:
            return 28
        elif key in {
            ("0", "0"),
            ("1", "0"),
            ("2", "0"),
            ("3", "0"),
            ("4", "0"),
            ("<s>", "0"),
        }:
            return 4
        elif key in {
            ("0", "2"),
            ("1", "2"),
            ("2", "2"),
            ("3", "2"),
            ("4", "2"),
            ("<s>", "2"),
        }:
            return 19
        elif key in {
            ("0", "1"),
            ("1", "1"),
            ("2", "1"),
            ("3", "1"),
            ("4", "1"),
            ("<s>", "1"),
        }:
            return 24
        elif key in {("</s>", "4")}:
            return 27
        elif key in {("</s>", "3")}:
            return 10
        return 1

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_3_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_7_output):
        key = num_attn_0_7_output
        return 5

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_7_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_4_output, num_attn_0_1_output):
        key = (num_attn_0_4_output, num_attn_0_1_output)
        return 0

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 2, 15, 17, 22}:
            return token == "0"
        elif mlp_0_1_output in {24, 1, 28}:
            return token == "2"
        elif mlp_0_1_output in {3, 16, 23, 25, 26, 27}:
            return token == "1"
        elif mlp_0_1_output in {10, 4}:
            return token == "</s>"
        elif mlp_0_1_output in {5, 6, 7, 8, 9, 11, 12, 14}:
            return token == ""
        elif mlp_0_1_output in {21, 20, 13}:
            return token == "4"
        elif mlp_0_1_output in {18, 19, 29}:
            return token == "3"

    attn_1_0_pattern = select_closest(tokens, mlp_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_2_output, token):
        if attn_0_2_output in {"4", "2", "<s>", "3", "0"}:
            return token == "1"
        elif attn_0_2_output in {"1"}:
            return token == "4"
        elif attn_0_2_output in {"</s>"}:
            return token == ""

    attn_1_1_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_2_output, token):
        if attn_0_2_output in {"<s>", "0"}:
            return token == "4"
        elif attn_0_2_output in {"1", "</s>"}:
            return token == ""
        elif attn_0_2_output in {"2"}:
            return token == "0"
        elif attn_0_2_output in {"3"}:
            return token == "2"
        elif attn_0_2_output in {"4"}:
            return token == "3"

    attn_1_2_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_2_output, position):
        if attn_0_2_output in {"0"}:
            return position == 5
        elif attn_0_2_output in {"1"}:
            return position == 14
        elif attn_0_2_output in {"2"}:
            return position == 18
        elif attn_0_2_output in {"3"}:
            return position == 6
        elif attn_0_2_output in {"4"}:
            return position == 17
        elif attn_0_2_output in {"</s>"}:
            return position == 16
        elif attn_0_2_output in {"<s>"}:
            return position == 4

    attn_1_3_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, token):
        if position in {0, 8, 13, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {1}:
            return token == "0"
        elif position in {17, 2, 15}:
            return token == "<s>"
        elif position in {11, 9, 3, 12}:
            return token == "2"
        elif position in {4, 5}:
            return token == "3"
        elif position in {6, 7}:
            return token == "4"
        elif position in {16, 10, 19, 14}:
            return token == "</s>"

    attn_1_4_pattern = select_closest(tokens, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_2_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0, 16, 18, 17}:
            return k_position == 1
        elif q_position in {1, 19}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 16
        elif q_position in {3}:
            return k_position == 14
        elif q_position in {9, 10, 4}:
            return k_position == 8
        elif q_position in {11, 5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {12, 13, 7}:
            return k_position == 5
        elif q_position in {8, 14}:
            return k_position == 4
        elif q_position in {15}:
            return k_position == 3
        elif q_position in {26, 20, 28}:
            return k_position == 18
        elif q_position in {25, 21}:
            return k_position == 23
        elif q_position in {22}:
            return k_position == 19
        elif q_position in {23}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 29
        elif q_position in {27}:
            return k_position == 24
        elif q_position in {29}:
            return k_position == 22

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_2_output, token):
        if attn_0_2_output in {"<s>", "1", "0", "2"}:
            return token == "3"
        elif attn_0_2_output in {"3"}:
            return token == "1"
        elif attn_0_2_output in {"</s>", "4"}:
            return token == ""

    attn_1_6_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, token):
        if position in {0}:
            return token == "4"
        elif position in {1, 12, 15, 17, 18}:
            return token == "<s>"
        elif position in {16, 2, 14}:
            return token == "</s>"
        elif position in {3}:
            return token == "0"
        elif position in {4, 5, 6, 7, 8, 9, 10, 11}:
            return token == "1"
        elif position in {19, 13}:
            return token == "2"
        elif position in {20, 23, 26, 28, 29}:
            return token == "3"
        elif position in {21, 22, 24, 25, 27}:
            return token == ""

    attn_1_7_pattern = select_closest(tokens, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_7_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_5_output, attn_0_4_output):
        if attn_0_5_output in {"0"}:
            return attn_0_4_output == "<pad>"
        elif attn_0_5_output in {"</s>", "2", "<s>", "3", "1"}:
            return attn_0_4_output == ""
        elif attn_0_5_output in {"4"}:
            return attn_0_4_output == "4"

    num_attn_1_0_pattern = select(attn_0_4_outputs, attn_0_5_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_5_output, attn_0_4_output):
        if attn_0_5_output in {"0"}:
            return attn_0_4_output == "0"
        elif attn_0_5_output in {"4", "2", "<s>", "3", "1"}:
            return attn_0_4_output == ""
        elif attn_0_5_output in {"</s>"}:
            return attn_0_4_output == "<pad>"

    num_attn_1_1_pattern = select(attn_0_4_outputs, attn_0_5_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_2_output, attn_0_5_output):
        if attn_0_2_output in {"</s>", "4", "2", "3", "0"}:
            return attn_0_5_output == ""
        elif attn_0_2_output in {"<s>", "1"}:
            return attn_0_5_output == "1"

    num_attn_1_2_pattern = select(attn_0_5_outputs, attn_0_2_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_4_output, attn_0_0_output):
        if attn_0_4_output in {"4", "2", "<s>", "1", "0"}:
            return attn_0_0_output == ""
        elif attn_0_4_output in {"3", "</s>"}:
            return attn_0_0_output == "</s>"

    num_attn_1_3_pattern = select(attn_0_0_outputs, attn_0_4_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_5_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_5_output, attn_0_3_output):
        if attn_0_5_output in {"</s>", "4", "<s>", "3", "1", "0"}:
            return attn_0_3_output == ""
        elif attn_0_5_output in {"2"}:
            return attn_0_3_output == "</s>"

    num_attn_1_4_pattern = select(attn_0_3_outputs, attn_0_5_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_5_output, attn_0_1_output):
        if attn_0_5_output in {"</s>", "4", "<s>", "3", "0"}:
            return attn_0_1_output == ""
        elif attn_0_5_output in {"1"}:
            return attn_0_1_output == "<s>"
        elif attn_0_5_output in {"2"}:
            return attn_0_1_output == "<pad>"

    num_attn_1_5_pattern = select(attn_0_1_outputs, attn_0_5_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(q_attn_0_7_output, k_attn_0_7_output):
        if q_attn_0_7_output in {"</s>", "4", "2", "1", "0"}:
            return k_attn_0_7_output == ""
        elif q_attn_0_7_output in {"<s>", "3"}:
            return k_attn_0_7_output == "3"

    num_attn_1_6_pattern = select(attn_0_7_outputs, attn_0_7_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_4_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_2_output, attn_0_5_output):
        if attn_0_2_output in {"</s>", "1", "0", "2"}:
            return attn_0_5_output == ""
        elif attn_0_2_output in {"<s>", "3"}:
            return attn_0_5_output == "3"
        elif attn_0_2_output in {"4"}:
            return attn_0_5_output == "<pad>"

    num_attn_1_7_pattern = select(attn_0_5_outputs, attn_0_2_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_1_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_6_output, attn_1_2_output):
        key = (attn_1_6_output, attn_1_2_output)
        return 12

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_1_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_1_output, position):
        key = (num_mlp_0_1_output, position)
        return 3

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, positions)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 10

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 1

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_2_output, token):
        if attn_0_2_output in {"4", "2", "3", "1", "0"}:
            return token == "<s>"
        elif attn_0_2_output in {"</s>"}:
            return token == "4"
        elif attn_0_2_output in {"<s>"}:
            return token == "</s>"

    attn_2_0_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_5_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_2_output, token):
        if attn_0_2_output in {"4", "2", "3", "1", "0"}:
            return token == "<s>"
        elif attn_0_2_output in {"</s>"}:
            return token == ""
        elif attn_0_2_output in {"<s>"}:
            return token == "2"

    attn_2_1_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {"</s>", "<s>", "0", "2"}:
            return k_attn_0_0_output == ""
        elif q_attn_0_0_output in {"1"}:
            return k_attn_0_0_output == "0"
        elif q_attn_0_0_output in {"3"}:
            return k_attn_0_0_output == "<s>"
        elif q_attn_0_0_output in {"4"}:
            return k_attn_0_0_output == "</s>"

    attn_2_2_pattern = select_closest(attn_0_0_outputs, attn_0_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_4_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_5_output, token):
        if attn_0_5_output in {"0"}:
            return token == "0"
        elif attn_0_5_output in {"3", "1", "</s>"}:
            return token == ""
        elif attn_0_5_output in {"4", "2"}:
            return token == "4"
        elif attn_0_5_output in {"<s>"}:
            return token == "2"

    attn_2_3_pattern = select_closest(tokens, attn_0_5_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_0_output, attn_0_2_output):
        if attn_0_0_output in {"0"}:
            return attn_0_2_output == "4"
        elif attn_0_0_output in {"<s>", "1"}:
            return attn_0_2_output == "0"
        elif attn_0_0_output in {"</s>", "2"}:
            return attn_0_2_output == ""
        elif attn_0_0_output in {"3"}:
            return attn_0_2_output == "3"
        elif attn_0_0_output in {"4"}:
            return attn_0_2_output == "2"

    attn_2_4_pattern = select_closest(attn_0_2_outputs, attn_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_3_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_7_output, token):
        if attn_0_7_output in {"2", "<s>", "3", "1", "0"}:
            return token == "<s>"
        elif attn_0_7_output in {"</s>", "4"}:
            return token == ""

    attn_2_5_pattern = select_closest(tokens, attn_0_7_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_2_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_2_output, position):
        if attn_0_2_output in {"4", "1", "0", "2"}:
            return position == 3
        elif attn_0_2_output in {"3"}:
            return position == 19
        elif attn_0_2_output in {"</s>"}:
            return position == 0
        elif attn_0_2_output in {"<s>"}:
            return position == 13

    attn_2_6_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_6_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "3"
        elif attn_0_0_output in {"1"}:
            return token == "4"
        elif attn_0_0_output in {"<s>", "3", "2"}:
            return token == ""
        elif attn_0_0_output in {"4"}:
            return token == "<s>"
        elif attn_0_0_output in {"</s>"}:
            return token == "</s>"

    attn_2_7_pattern = select_closest(tokens, attn_0_0_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_1_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_5_output, attn_0_0_output):
        if attn_1_5_output in {"</s>", "4", "<s>", "3", "1", "0"}:
            return attn_0_0_output == ""
        elif attn_1_5_output in {"2"}:
            return attn_0_0_output == "</s>"

    num_attn_2_0_pattern = select(attn_0_0_outputs, attn_1_5_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_2_output, attn_0_5_output):
        if attn_0_2_output in {"</s>", "4", "3", "1", "0"}:
            return attn_0_5_output == ""
        elif attn_0_2_output in {"<s>", "2"}:
            return attn_0_5_output == "2"

    num_attn_2_1_pattern = select(attn_0_5_outputs, attn_0_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_2_output, attn_0_5_output):
        if attn_0_2_output in {"<s>", "0"}:
            return attn_0_5_output == "0"
        elif attn_0_2_output in {"</s>", "4", "2", "3", "1"}:
            return attn_0_5_output == ""

    num_attn_2_2_pattern = select(attn_0_5_outputs, attn_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_2_output, attn_0_5_output):
        if attn_0_2_output in {"</s>", "2", "3", "1", "0"}:
            return attn_0_5_output == ""
        elif attn_0_2_output in {"<s>", "4"}:
            return attn_0_5_output == "4"

    num_attn_2_3_pattern = select(attn_0_5_outputs, attn_0_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_5_output, mlp_0_0_output):
        if attn_0_5_output in {"0"}:
            return mlp_0_0_output == 15
        elif attn_0_5_output in {"<s>", "1", "</s>"}:
            return mlp_0_0_output == 11
        elif attn_0_5_output in {"2"}:
            return mlp_0_0_output == 12
        elif attn_0_5_output in {"3"}:
            return mlp_0_0_output == 25
        elif attn_0_5_output in {"4"}:
            return mlp_0_0_output == 19

    num_attn_2_4_pattern = select(mlp_0_0_outputs, attn_0_5_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_2_output, attn_0_5_output):
        if attn_0_2_output in {"</s>", "4", "<s>", "3", "1", "0"}:
            return attn_0_5_output == ""
        elif attn_0_2_output in {"2"}:
            return attn_0_5_output == "0"

    num_attn_2_5_pattern = select(attn_0_5_outputs, attn_0_2_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_3_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_5_output, attn_0_0_output):
        if attn_1_5_output in {"0"}:
            return attn_0_0_output == "<pad>"
        elif attn_1_5_output in {"</s>", "4", "2", "<s>", "1"}:
            return attn_0_0_output == ""
        elif attn_1_5_output in {"3"}:
            return attn_0_0_output == "</s>"

    num_attn_2_6_pattern = select(attn_0_0_outputs, attn_1_5_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_0_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(q_attn_0_5_output, k_attn_0_5_output):
        if q_attn_0_5_output in {"</s>", "4", "<s>", "3", "1", "0"}:
            return k_attn_0_5_output == ""
        elif q_attn_0_5_output in {"2"}:
            return k_attn_0_5_output == "4"

    num_attn_2_7_pattern = select(attn_0_5_outputs, attn_0_5_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_1_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, num_mlp_0_0_output):
        key = (attn_2_3_output, num_mlp_0_0_output)
        return 15

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, num_mlp_0_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_0_1_output, attn_2_2_output):
        key = (num_mlp_0_1_output, attn_2_2_output)
        return 10

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_2_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 13

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_1_6_output):
        key = (num_attn_2_3_output, num_attn_1_6_output)
        return 14

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_6_outputs)
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


print(run(["<s>", "0", "3", "2", "3", "0", "2", "1", "3", "2", "</s>"]))
