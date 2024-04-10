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
        "output/length/rasp/reverse/trainlength10/s4/reverse_weights.csv",
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
            return k_position == 3
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {8, 9, 6, 7}:
            return k_position == 1
        elif q_position in {10, 18}:
            return k_position == 14
        elif q_position in {16, 19, 11, 14}:
            return k_position == 16
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {13}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 17

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 2
        elif q_position in {9, 7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 1
        elif q_position in {10, 12}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {18, 13}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {16, 15}:
            return k_position == 17
        elif q_position in {17, 19}:
            return k_position == 16

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 3}:
            return token == "0"
        elif position in {8, 1, 2}:
            return token == "1"
        elif position in {4}:
            return token == "4"
        elif position in {19, 15, 5, 7}:
            return token == "3"
        elif position in {9, 6}:
            return token == "</s>"
        elif position in {10, 11, 12, 13, 14, 16, 17, 18}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8, 9}:
            return k_position == 1
        elif q_position in {10, 18}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 13
        elif q_position in {16}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {19}:
            return k_position == 10

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {8, 1}:
            return k_position == 5
        elif q_position in {2, 3}:
            return k_position == 0
        elif q_position in {4, 6}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 6
        elif q_position in {10, 18, 13}:
            return k_position == 12
        elif q_position in {11, 12}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {16, 19}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 16

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2, 3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 0
        elif q_position in {5, 7}:
            return k_position == 2
        elif q_position in {8, 9}:
            return k_position == 1
        elif q_position in {10, 14}:
            return k_position == 11
        elif q_position in {16, 11, 13}:
            return k_position == 17
        elif q_position in {12, 15}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 10
        elif q_position in {19}:
            return k_position == 18

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 2}:
            return k_position == 6
        elif q_position in {3, 4, 5}:
            return k_position == 0
        elif q_position in {6}:
            return k_position == 2
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8, 9}:
            return k_position == 1
        elif q_position in {16, 10, 15}:
            return k_position == 13
        elif q_position in {17, 11}:
            return k_position == 18
        elif q_position in {12, 13}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 11
        elif q_position in {18}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 12

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 9}:
            return k_position == 6
        elif q_position in {2, 3}:
            return k_position == 0
        elif q_position in {4, 6}:
            return k_position == 9
        elif q_position in {8, 5, 7}:
            return k_position == 1
        elif q_position in {16, 10, 15}:
            return k_position == 17
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {18, 13}:
            return k_position == 18
        elif q_position in {14}:
            return k_position == 11
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 19

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"</s>", "0", "2", "4", "<s>", "1", "3"}:
            return position == 9

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"1", "0"}:
            return position == 8
        elif token in {"2"}:
            return position == 14
        elif token in {"4", "3"}:
            return position == 7
        elif token in {"</s>"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 6

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"0"}:
            return position == 17
        elif token in {"1", "4"}:
            return position == 7
        elif token in {"2"}:
            return position == 9
        elif token in {"3"}:
            return position == 8
        elif token in {"</s>"}:
            return position == 14
        elif token in {"<s>"}:
            return position == 0

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"</s>", "0", "2", "4", "<s>", "1", "3"}:
            return position == 9

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"2", "0", "4"}:
            return position == 8
        elif token in {"1", "3"}:
            return position == 7
        elif token in {"</s>"}:
            return position == 17
        elif token in {"<s>"}:
            return position == 6

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"1", "0", "4", "3"}:
            return position == 8
        elif token in {"2"}:
            return position == 7
        elif token in {"</s>"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 0

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0, 2, 3, 4, 5, 6, 7}:
            return k_position == 8
        elif q_position in {1, 9}:
            return k_position == 7
        elif q_position in {8, 17, 11, 15}:
            return k_position == 0
        elif q_position in {10}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 2
        elif q_position in {18, 13}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {16}:
            return k_position == 18
        elif q_position in {19}:
            return k_position == 16

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"2", "0"}:
            return position == 8
        elif token in {"</s>", "1", "3"}:
            return position == 9
        elif token in {"4"}:
            return position == 18
        elif token in {"<s>"}:
            return position == 6

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_6_output):
        key = attn_0_6_output
        return 6

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_6_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, attn_0_6_output):
        key = (attn_0_0_output, attn_0_6_output)
        if key in {("0", "</s>"), ("3", "</s>")}:
            return 6
        elif key in {
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "<s>"),
            ("4", "3"),
            ("</s>", "3"),
            ("<s>", "3"),
        }:
            return 8
        return 14

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_6_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_3_output):
        key = (num_attn_0_5_output, num_attn_0_3_output)
        return 0

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_1_output):
        key = (num_attn_0_3_output, num_attn_0_1_output)
        return 5

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 4, 7}:
            return k_position == 2
        elif q_position in {1, 6}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 0
        elif q_position in {3}:
            return k_position == 9
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {8}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {10, 14}:
            return k_position == 16
        elif q_position in {16, 11, 15}:
            return k_position == 18
        elif q_position in {17, 12}:
            return k_position == 10
        elif q_position in {19, 13}:
            return k_position == 14
        elif q_position in {18}:
            return k_position == 13

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {8, 3}:
            return k_position == 3
        elif q_position in {4, 5}:
            return k_position == 7
        elif q_position in {9, 6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {10, 13, 15}:
            return k_position == 15
        elif q_position in {11, 14}:
            return k_position == 17
        elif q_position in {17, 12}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 13
        elif q_position in {18, 19}:
            return k_position == 18

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {9, 6, 17}:
            return k_position == 8
        elif q_position in {16, 19, 7}:
            return k_position == 9
        elif q_position in {10}:
            return k_position == 18
        elif q_position in {11, 13}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {18, 15}:
            return k_position == 13

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0, 8}:
            return token == "2"
        elif position in {1, 6}:
            return token == "0"
        elif position in {2, 4}:
            return token == "3"
        elif position in {3, 5}:
            return token == "1"
        elif position in {7}:
            return token == "</s>"
        elif position in {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == ""

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {8, 3}:
            return k_position == 3
        elif q_position in {4, 7}:
            return k_position == 7
        elif q_position in {9, 5, 6}:
            return k_position == 8
        elif q_position in {10, 18}:
            return k_position == 12
        elif q_position in {11, 13}:
            return k_position == 14
        elif q_position in {16, 12}:
            return k_position == 18
        elif q_position in {17, 14}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 11
        elif q_position in {19}:
            return k_position == 17

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0, 9}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {5, 7}:
            return k_position == 0
        elif q_position in {8, 6}:
            return k_position == 1
        elif q_position in {10, 11}:
            return k_position == 17
        elif q_position in {16, 12, 15}:
            return k_position == 16
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 13
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 18

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 9}:
            return k_position == 8
        elif q_position in {8, 1}:
            return k_position == 9
        elif q_position in {2, 4}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {5, 7}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {10, 19}:
            return k_position == 15
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {18, 12}:
            return k_position == 16
        elif q_position in {13, 14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 12
        elif q_position in {16}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 18

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_0_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_position, k_position):
        if q_position in {0, 3, 5}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 6}:
            return k_position == 7
        elif q_position in {8, 4}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 0
        elif q_position in {9}:
            return k_position == 9
        elif q_position in {10, 12, 15}:
            return k_position == 16
        elif q_position in {11, 13}:
            return k_position == 18
        elif q_position in {16, 17, 14}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 12

    attn_1_7_pattern = select_closest(positions, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_4_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"1", "0", "3"}:
            return k_attn_0_1_output == "4"
        elif q_attn_0_1_output in {"2"}:
            return k_attn_0_1_output == "2"
        elif q_attn_0_1_output in {"4"}:
            return k_attn_0_1_output == "</s>"
        elif q_attn_0_1_output in {"</s>"}:
            return k_attn_0_1_output == "1"
        elif q_attn_0_1_output in {"<s>"}:
            return k_attn_0_1_output == ""

    num_attn_1_0_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_6_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"2", "0"}:
            return k_attn_0_1_output == "1"
        elif q_attn_0_1_output in {"</s>", "1", "4"}:
            return k_attn_0_1_output == ""
        elif q_attn_0_1_output in {"3"}:
            return k_attn_0_1_output == "3"
        elif q_attn_0_1_output in {"<s>"}:
            return k_attn_0_1_output == "<s>"

    num_attn_1_1_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_6_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "4"
        elif attn_0_3_output in {"</s>", "2", "4", "<s>", "1"}:
            return token == ""
        elif attn_0_3_output in {"3"}:
            return token == "<pad>"

    num_attn_1_2_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"0", "4", "3"}:
            return attn_0_1_output == ""
        elif attn_0_3_output in {"2", "1"}:
            return attn_0_1_output == "</s>"
        elif attn_0_3_output in {"</s>"}:
            return attn_0_1_output == "2"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_1_output == "1"

    num_attn_1_3_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_1_output, attn_0_6_output):
        if attn_0_1_output in {"0", "4", "<s>", "1", "3"}:
            return attn_0_6_output == ""
        elif attn_0_1_output in {"2"}:
            return attn_0_6_output == "<s>"
        elif attn_0_1_output in {"</s>"}:
            return attn_0_6_output == "2"

    num_attn_1_4_pattern = select(attn_0_6_outputs, attn_0_1_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_4_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"</s>", "0", "<s>", "1", "3"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"2"}:
            return k_attn_0_3_output == "2"
        elif q_attn_0_3_output in {"4"}:
            return k_attn_0_3_output == "3"

    num_attn_1_5_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"2", "<s>", "1", "0"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"3"}:
            return k_attn_0_3_output == "3"
        elif q_attn_0_3_output in {"4"}:
            return k_attn_0_3_output == "4"
        elif q_attn_0_3_output in {"</s>"}:
            return k_attn_0_3_output == "<s>"

    num_attn_1_6_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_3_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"2", "0"}:
            return k_attn_0_1_output == "3"
        elif q_attn_0_1_output in {"</s>", "<s>", "1", "3"}:
            return k_attn_0_1_output == ""
        elif q_attn_0_1_output in {"4"}:
            return k_attn_0_1_output == "4"

    num_attn_1_7_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_6_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_1_output, attn_1_5_output):
        key = (attn_0_1_output, attn_1_5_output)
        if key in {
            ("0", "0"),
            ("0", "3"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "3"),
            ("2", "0"),
            ("2", "3"),
            ("3", "0"),
            ("3", "3"),
            ("4", "0"),
            ("4", "3"),
            ("<s>", "0"),
            ("<s>", "3"),
        }:
            return 8
        elif key in {
            ("0", "</s>"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("</s>", "0"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 7
        elif key in {("0", "2"), ("1", "2"), ("2", "2"), ("2", "</s>"), ("2", "<s>")}:
            return 4
        elif key in {
            ("0", "1"),
            ("1", "1"),
            ("2", "1"),
            ("3", "1"),
            ("4", "1"),
            ("</s>", "1"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("<s>", "1"),
        }:
            return 6
        elif key in {
            ("0", "4"),
            ("1", "4"),
            ("2", "4"),
            ("3", "4"),
            ("4", "4"),
            ("<s>", "4"),
        }:
            return 18
        return 13

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_1_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_6_output, attn_0_7_output):
        key = (attn_1_6_output, attn_0_7_output)
        return 5

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_0_7_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_1_2_output):
        key = (num_attn_1_3_output, num_attn_1_2_output)
        return 19

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_4_output, num_attn_1_3_output):
        key = (num_attn_1_4_output, num_attn_1_3_output)
        if key in {
            (6, 1),
            (7, 1),
            (8, 1),
            (9, 1),
            (10, 1),
            (11, 1),
            (12, 0),
            (12, 1),
            (13, 0),
            (13, 1),
            (14, 0),
            (14, 1),
            (15, 0),
            (15, 1),
            (16, 0),
            (16, 1),
            (16, 2),
            (17, 0),
            (17, 1),
            (17, 2),
            (18, 0),
            (18, 1),
            (18, 2),
            (19, 0),
            (19, 1),
            (19, 2),
            (20, 0),
            (20, 1),
            (20, 2),
            (21, 0),
            (21, 1),
            (21, 2),
            (22, 0),
            (22, 1),
            (22, 2),
            (23, 0),
            (23, 1),
            (23, 2),
            (24, 0),
            (24, 1),
            (24, 2),
            (25, 0),
            (25, 1),
            (25, 2),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
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
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
        }:
            return 10
        elif key in {
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
        }:
            return 12
        return 6

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_1_output, position):
        if attn_0_1_output in {"0"}:
            return position == 7
        elif attn_0_1_output in {"2", "4", "<s>", "1", "3"}:
            return position == 3
        elif attn_0_1_output in {"</s>"}:
            return position == 1

    attn_2_0_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_7_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 3, 10, 13, 14, 16, 17, 19}:
            return token == "2"
        elif position in {1, 4, 6}:
            return token == "</s>"
        elif position in {2, 7, 9, 11, 12, 15, 18}:
            return token == "0"
        elif position in {5}:
            return token == "4"
        elif position in {8}:
            return token == "<s>"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_5_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 2, 10, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "0"
        elif position in {1}:
            return token == "2"
        elif position in {3, 4, 7, 8, 11}:
            return token == "1"
        elif position in {5}:
            return token == "3"
        elif position in {6}:
            return token == "4"
        elif position in {9}:
            return token == ""

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 9
        elif q_position in {8, 3, 7}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 1

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_4_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_6_output, token):
        if attn_0_6_output in {"2", "0", "4", "3"}:
            return token == "3"
        elif attn_0_6_output in {"1"}:
            return token == ""
        elif attn_0_6_output in {"</s>"}:
            return token == "2"
        elif attn_0_6_output in {"<s>"}:
            return token == "</s>"

    attn_2_4_pattern = select_closest(tokens, attn_0_6_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_0_2_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"<s>", "</s>"}:
            return k_token == ""

    attn_2_5_pattern = select_closest(tokens, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_4_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(position, token):
        if position in {0, 3, 4, 7, 10, 13, 14, 17, 18, 19}:
            return token == "2"
        elif position in {1, 2, 6}:
            return token == "0"
        elif position in {8, 5}:
            return token == "</s>"
        elif position in {9, 11, 12, 15, 16}:
            return token == ""

    attn_2_6_pattern = select_closest(tokens, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, tokens)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(position, token):
        if position in {0, 6, 10, 11, 12, 13, 14, 15, 16, 18, 19}:
            return token == "0"
        elif position in {1}:
            return token == "2"
        elif position in {2, 5}:
            return token == "</s>"
        elif position in {3, 4, 7}:
            return token == "1"
        elif position in {8}:
            return token == "4"
        elif position in {9, 17}:
            return token == ""

    attn_2_7_pattern = select_closest(tokens, positions, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_0_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"0"}:
            return k_attn_0_1_output == "0"
        elif q_attn_0_1_output in {"2", "1", "</s>"}:
            return k_attn_0_1_output == "</s>"
        elif q_attn_0_1_output in {"4", "3"}:
            return k_attn_0_1_output == "1"
        elif q_attn_0_1_output in {"<s>"}:
            return k_attn_0_1_output == ""

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_6_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0"}:
            return k_attn_0_3_output == "0"
        elif q_attn_0_3_output in {"</s>", "2", "4", "<s>", "1", "3"}:
            return k_attn_0_3_output == ""

    num_attn_2_1_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"<s>", "</s>", "0", "3"}:
            return attn_0_1_output == ""
        elif attn_0_3_output in {"1", "4"}:
            return attn_0_1_output == "</s>"
        elif attn_0_3_output in {"2"}:
            return attn_0_1_output == "<pad>"

    num_attn_2_2_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"</s>", "0"}:
            return k_attn_0_1_output == ""
        elif q_attn_0_1_output in {"1", "3"}:
            return k_attn_0_1_output == "2"
        elif q_attn_0_1_output in {"2"}:
            return k_attn_0_1_output == "</s>"
        elif q_attn_0_1_output in {"4"}:
            return k_attn_0_1_output == "4"
        elif q_attn_0_1_output in {"<s>"}:
            return k_attn_0_1_output == "<pad>"

    num_attn_2_3_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_6_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"0"}:
            return k_attn_0_1_output == "</s>"
        elif q_attn_0_1_output in {"1", "4", "3"}:
            return k_attn_0_1_output == "0"
        elif q_attn_0_1_output in {"2"}:
            return k_attn_0_1_output == "2"
        elif q_attn_0_1_output in {"</s>"}:
            return k_attn_0_1_output == ""
        elif q_attn_0_1_output in {"<s>"}:
            return k_attn_0_1_output == "<s>"

    num_attn_2_4_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_6_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_3_output, token):
        if attn_0_3_output in {"</s>", "0", "<s>", "1", "3"}:
            return token == ""
        elif attn_0_3_output in {"2"}:
            return token == "2"
        elif attn_0_3_output in {"4"}:
            return token == "4"

    num_attn_2_5_pattern = select(tokens, attn_0_3_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_0_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"2", "0", "4"}:
            return attn_0_1_output == ""
        elif attn_0_3_output in {"<s>", "1", "3"}:
            return attn_0_1_output == "</s>"
        elif attn_0_3_output in {"</s>"}:
            return attn_0_1_output == "2"

    num_attn_2_6_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_7_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"</s>", "0", "4", "<s>", "1"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"2"}:
            return k_attn_0_3_output == "1"
        elif q_attn_0_3_output in {"3"}:
            return k_attn_0_3_output == "4"

    num_attn_2_7_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_0_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_0_1_output, num_mlp_0_0_output):
        key = (mlp_0_1_output, num_mlp_0_0_output)
        return 18

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, num_mlp_0_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_6_output, attn_0_5_output):
        key = (attn_0_6_output, attn_0_5_output)
        if key in {("</s>", "4"), ("</s>", "<s>")}:
            return 8
        elif key in {("</s>", "0"), ("</s>", "1"), ("</s>", "2")}:
            return 3
        elif key in {("</s>", "3")}:
            return 13
        return 14

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_5_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_7_output, num_attn_2_1_output):
        key = (num_attn_1_7_output, num_attn_2_1_output)
        if key in {(0, 0)}:
            return 0
        return 8

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_5_output, num_attn_1_0_output):
        key = (num_attn_1_5_output, num_attn_1_0_output)
        return 1

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_0_outputs)
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


print(run(["<s>", "1", "0", "0", "</s>"]))
