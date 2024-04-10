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
        "output/length/rasp/dyck2/trainlength20/s2/dyck2_weights.csv",
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
        if q_position in {0, 21}:
            return k_position == 37
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4, 29}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
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
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20, 23}:
            return k_position == 21
        elif q_position in {22, 30}:
            return k_position == 35
        elif q_position in {24}:
            return k_position == 30
        elif q_position in {25, 28}:
            return k_position == 29
        elif q_position in {26, 31}:
            return k_position == 32
        elif q_position in {27}:
            return k_position == 20
        elif q_position in {32, 36, 37}:
            return k_position == 23
        elif q_position in {33}:
            return k_position == 36
        elif q_position in {34, 39}:
            return k_position == 34
        elif q_position in {35}:
            return k_position == 24
        elif q_position in {38}:
            return k_position == 39

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 33, 4, 36, 6, 37, 38, 23, 29}:
            return k_position == 5
        elif q_position in {32, 1, 2, 3, 5, 21, 22, 24, 25, 26, 27, 28}:
            return k_position == 2
        elif q_position in {7, 12, 14, 16, 18}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {17}:
            return k_position == 4
        elif q_position in {19}:
            return k_position == 1
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {30}:
            return k_position == 29
        elif q_position in {31}:
            return k_position == 28
        elif q_position in {34}:
            return k_position == 20
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {39}:
            return k_position == 36

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 23}:
            return k_position == 37
        elif q_position in {1, 18}:
            return k_position == 1
        elif q_position in {
            32,
            33,
            2,
            3,
            34,
            36,
            37,
            38,
            39,
            20,
            21,
            22,
            24,
            25,
            27,
            28,
            29,
            30,
        }:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 36
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8, 16}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10, 11}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {26}:
            return k_position == 22
        elif q_position in {31}:
            return k_position == 19
        elif q_position in {35}:
            return k_position == 24

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 37, 21}:
            return k_position == 33
        elif q_position in {1, 4}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 26
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {32, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
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
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20, 38, 39}:
            return k_position == 23
        elif q_position in {34, 36, 22}:
            return k_position == 22
        elif q_position in {35, 23}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 32
        elif q_position in {25}:
            return k_position == 38
        elif q_position in {26}:
            return k_position == 35
        elif q_position in {27}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 25
        elif q_position in {29}:
            return k_position == 30
        elif q_position in {30}:
            return k_position == 31
        elif q_position in {31}:
            return k_position == 34
        elif q_position in {33}:
            return k_position == 37

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 33
        elif token in {")"}:
            return position == 7
        elif token in {"<s>", "}"}:
            return position == 4
        elif token in {"{"}:
            return position == 18

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {
            0,
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
            37,
            38,
        }:
            return token == ""
        elif position in {1, 25}:
            return token == "<s>"
        elif position in {2, 3, 7}:
            return token == "{"
        elif position in {35, 36, 39}:
            return token == "("

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 35
        elif token in {")", "}"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 23
        elif token in {"{"}:
            return position == 29

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {")", "{", "("}:
            return k_token == ""
        elif q_token in {"<s>"}:
            return k_token == "("
        elif q_token in {"}"}:
            return k_token == "<s>"

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(token, attn_0_0_output):
        key = (token, attn_0_0_output)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("<s>", "("),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 4
        elif key in {
            (")", ")"),
            (")", "<s>"),
            (")", "{"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "{"),
            ("{", "}"),
            ("}", ")"),
            ("}", "{"),
            ("}", "}"),
        }:
            return 35
        elif key in {("(", ")"), (")", "("), ("{", ")")}:
            return 17
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(tokens, attn_0_0_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, token):
        key = (attn_0_3_output, token)
        if key in {
            ("(", "}"),
            ("<s>", "}"),
            ("{", "}"),
            ("}", "("),
            ("}", "<s>"),
            ("}", "{"),
            ("}", "}"),
        }:
            return 32
        elif key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 7
        elif key in {("(", ")"), (")", "("), (")", "<s>"), (")", "{")}:
            return 12
        elif key in {("<s>", ")")}:
            return 2
        elif key in {("{", ")")}:
            return 33
        return 35

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_0_output, token):
        key = (attn_0_0_output, token)
        if key in {("(", "("), ("(", "<s>"), ("(", "{"), ("(", "}")}:
            return 14
        elif key in {
            ("(", ")"),
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            (")", "{"),
            (")", "}"),
            ("}", ")"),
        }:
            return 21
        elif key in {("{", "("), ("{", ")"), ("{", "<s>"), ("{", "{")}:
            return 6
        elif key in {("<s>", ")")}:
            return 5
        return 32

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_0_outputs, tokens)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_3_output, attn_0_1_output):
        key = (attn_0_3_output, attn_0_1_output)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            (")", "("),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 4
        elif key in {("(", "}"), ("}", "("), ("}", "<s>"), ("}", "{"), ("}", "}")}:
            return 5
        elif key in {("<s>", "("), ("<s>", "<s>"), ("<s>", "{")}:
            return 7
        elif key in {("<s>", ")")}:
            return 2
        elif key in {("<s>", "}")}:
            return 38
        return 35

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_1_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        return 10

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 39

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 16

    num_mlp_0_2_outputs = [num_mlp_0_2(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_0_output, num_attn_0_2_output):
        key = (num_attn_0_0_output, num_attn_0_2_output)
        return 30

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_3_output, position):
        if attn_0_3_output in {"<s>", "("}:
            return position == 1
        elif attn_0_3_output in {")"}:
            return position == 5
        elif attn_0_3_output in {"{"}:
            return position == 4
        elif attn_0_3_output in {"}"}:
            return position == 3

    attn_1_0_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_3_output, mlp_0_2_output):
        if attn_0_3_output in {")", "{", "("}:
            return mlp_0_2_output == 4
        elif attn_0_3_output in {"<s>"}:
            return mlp_0_2_output == 5
        elif attn_0_3_output in {"}"}:
            return mlp_0_2_output == 6

    attn_1_1_pattern = select_closest(mlp_0_2_outputs, attn_0_3_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, position):
        if token in {"{", "("}:
            return position == 3
        elif token in {")", "<s>", "}"}:
            return position == 4

    attn_1_2_pattern = select_closest(positions, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"("}:
            return position == 6
        elif token in {")", "}"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 4
        elif token in {"{"}:
            return position == 5

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_2_output, token):
        if attn_0_2_output in {"("}:
            return token == ")"
        elif attn_0_2_output in {")", "<s>", "}"}:
            return token == ""
        elif attn_0_2_output in {"{"}:
            return token == "}"

    num_attn_1_0_pattern = select(tokens, attn_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, attn_0_0_output):
        if attn_0_3_output in {"<s>", "{", "("}:
            return attn_0_0_output == ""
        elif attn_0_3_output in {")"}:
            return attn_0_0_output == "("
        elif attn_0_3_output in {"}"}:
            return attn_0_0_output == "{"

    num_attn_1_1_pattern = select(attn_0_0_outputs, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, attn_0_0_output):
        if position in {0, 2, 3, 35, 5, 6, 36, 37, 12, 16, 18, 22, 23, 24, 29}:
            return attn_0_0_output == ")"
        elif position in {1, 4, 38, 7, 39, 11, 13, 15, 17, 19, 27, 28, 30}:
            return attn_0_0_output == ""
        elif position in {32, 33, 34, 8, 9, 10, 14, 20, 21, 25, 26, 31}:
            return attn_0_0_output == "}"

    num_attn_1_2_pattern = select(attn_0_0_outputs, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_0_output):
        if position in {0, 1, 4, 7, 13, 15, 17, 19, 27}:
            return attn_0_0_output == ""
        elif position in {32, 33, 2, 34, 5, 6, 38, 12, 16, 18, 21, 22, 24, 25, 29, 31}:
            return attn_0_0_output == "}"
        elif position in {3, 35, 36, 37, 39, 8, 9, 10, 11, 14, 20, 23, 26, 28, 30}:
            return attn_0_0_output == ")"

    num_attn_1_3_pattern = select(attn_0_0_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, attn_1_0_output):
        key = (attn_1_2_output, attn_1_0_output)
        if key in {
            (3, 3),
            (6, 3),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 11),
            (11, 12),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 18),
            (11, 19),
            (11, 20),
            (11, 21),
            (11, 22),
            (11, 25),
            (11, 26),
            (11, 27),
            (11, 28),
            (11, 30),
            (11, 31),
            (11, 32),
            (11, 33),
            (11, 35),
            (11, 36),
            (11, 37),
            (11, 39),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 11),
            (13, 12),
            (13, 13),
            (13, 14),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (13, 25),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 30),
            (13, 31),
            (13, 32),
            (13, 33),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 39),
            (16, 3),
            (22, 3),
            (26, 3),
            (36, 3),
        }:
            return 22
        elif key in {
            (0, 9),
            (1, 9),
            (1, 31),
            (3, 8),
            (3, 9),
            (3, 31),
            (15, 8),
            (15, 9),
            (15, 17),
            (15, 19),
            (15, 31),
            (16, 9),
            (17, 9),
            (17, 31),
            (20, 9),
            (24, 9),
            (26, 9),
            (27, 9),
            (29, 9),
            (30, 9),
            (30, 31),
            (31, 8),
            (31, 9),
            (31, 17),
            (31, 31),
            (32, 9),
            (33, 9),
            (34, 9),
        }:
            return 23
        return 31

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_0_output, attn_1_3_output):
        key = (attn_1_0_output, attn_1_3_output)
        if key in {
            (0, 4),
            (0, 7),
            (0, 8),
            (0, 33),
            (1, 4),
            (1, 7),
            (1, 8),
            (1, 14),
            (1, 18),
            (1, 20),
            (1, 22),
            (1, 27),
            (1, 33),
            (2, 4),
            (2, 7),
            (2, 8),
            (3, 4),
            (3, 7),
            (3, 8),
            (3, 14),
            (3, 33),
            (4, 7),
            (4, 8),
            (5, 7),
            (5, 8),
            (6, 7),
            (7, 0),
            (7, 4),
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
            (7, 11),
            (7, 13),
            (7, 14),
            (7, 18),
            (7, 19),
            (7, 20),
            (7, 22),
            (7, 23),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 29),
            (7, 30),
            (7, 33),
            (7, 36),
            (7, 37),
            (7, 39),
            (8, 4),
            (8, 7),
            (8, 8),
            (8, 14),
            (8, 18),
            (8, 20),
            (8, 22),
            (8, 27),
            (8, 33),
            (9, 4),
            (9, 7),
            (9, 8),
            (9, 14),
            (9, 33),
            (10, 4),
            (10, 7),
            (10, 8),
            (10, 33),
            (11, 4),
            (11, 7),
            (11, 8),
            (11, 14),
            (11, 33),
            (12, 7),
            (13, 4),
            (13, 7),
            (13, 8),
            (13, 14),
            (13, 33),
            (14, 4),
            (14, 7),
            (14, 8),
            (15, 4),
            (15, 7),
            (15, 8),
            (15, 33),
            (16, 7),
            (16, 8),
            (17, 4),
            (17, 7),
            (17, 8),
            (17, 14),
            (17, 33),
            (18, 4),
            (18, 7),
            (18, 8),
            (19, 4),
            (19, 7),
            (19, 8),
            (19, 13),
            (19, 14),
            (19, 18),
            (19, 20),
            (19, 22),
            (19, 23),
            (19, 27),
            (19, 33),
            (19, 39),
            (20, 4),
            (20, 7),
            (20, 8),
            (21, 7),
            (21, 8),
            (22, 4),
            (22, 7),
            (22, 8),
            (22, 33),
            (23, 4),
            (23, 7),
            (23, 8),
            (24, 4),
            (24, 7),
            (24, 8),
            (25, 4),
            (25, 7),
            (25, 8),
            (26, 4),
            (26, 7),
            (26, 8),
            (26, 33),
            (27, 4),
            (27, 7),
            (27, 8),
            (27, 33),
            (28, 7),
            (28, 8),
            (29, 4),
            (29, 7),
            (29, 8),
            (29, 33),
            (30, 4),
            (30, 7),
            (30, 8),
            (31, 7),
            (32, 7),
            (32, 8),
            (33, 4),
            (33, 7),
            (33, 8),
            (33, 33),
            (34, 0),
            (34, 4),
            (34, 6),
            (34, 7),
            (34, 8),
            (34, 9),
            (34, 11),
            (34, 13),
            (34, 14),
            (34, 15),
            (34, 17),
            (34, 18),
            (34, 19),
            (34, 20),
            (34, 22),
            (34, 23),
            (34, 24),
            (34, 25),
            (34, 26),
            (34, 27),
            (34, 28),
            (34, 29),
            (34, 30),
            (34, 33),
            (34, 36),
            (34, 37),
            (34, 38),
            (34, 39),
            (35, 4),
            (35, 7),
            (35, 8),
            (35, 33),
            (36, 4),
            (36, 7),
            (36, 8),
            (37, 4),
            (37, 7),
            (37, 8),
            (38, 7),
            (39, 4),
            (39, 7),
            (39, 8),
            (39, 33),
        }:
            return 29
        elif key in {
            (2, 33),
            (14, 33),
            (18, 33),
            (23, 33),
            (24, 33),
            (30, 33),
            (33, 14),
            (33, 18),
            (34, 10),
        }:
            return 8
        return 34

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_1_output, attn_0_2_output):
        key = (attn_1_1_output, attn_0_2_output)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("(", "}"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("<s>", "}"),
            ("{", "("),
            ("{", "{"),
            ("{", "}"),
            ("}", "("),
            ("}", "}"),
        }:
            return 1
        return 30

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_0_2_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_3_output, num_mlp_0_1_output):
        key = (attn_1_3_output, num_mlp_0_1_output)
        if key in {
            (0, 0),
            (0, 17),
            (0, 24),
            (0, 28),
            (0, 29),
            (0, 33),
            (1, 24),
            (1, 29),
            (1, 33),
            (2, 24),
            (2, 29),
            (2, 33),
            (3, 33),
            (4, 33),
            (5, 33),
            (6, 17),
            (6, 24),
            (6, 29),
            (6, 33),
            (7, 33),
            (8, 29),
            (8, 33),
            (9, 33),
            (10, 0),
            (10, 17),
            (10, 24),
            (10, 28),
            (10, 29),
            (10, 33),
            (10, 34),
            (11, 33),
            (12, 0),
            (12, 17),
            (12, 24),
            (12, 28),
            (12, 29),
            (12, 33),
            (12, 34),
            (13, 0),
            (13, 17),
            (13, 24),
            (13, 28),
            (13, 29),
            (13, 33),
            (13, 34),
            (14, 0),
            (14, 17),
            (14, 24),
            (14, 29),
            (14, 33),
            (15, 33),
            (16, 29),
            (16, 33),
            (17, 0),
            (17, 17),
            (17, 24),
            (17, 28),
            (17, 29),
            (17, 33),
            (17, 34),
            (18, 33),
            (19, 0),
            (19, 10),
            (19, 17),
            (19, 22),
            (19, 24),
            (19, 28),
            (19, 29),
            (19, 31),
            (19, 33),
            (19, 34),
            (20, 0),
            (20, 17),
            (20, 24),
            (20, 29),
            (20, 33),
            (20, 34),
            (21, 33),
            (22, 33),
            (23, 33),
            (24, 33),
            (25, 33),
            (26, 33),
            (27, 33),
            (28, 33),
            (29, 0),
            (29, 17),
            (29, 24),
            (29, 28),
            (29, 29),
            (29, 33),
            (30, 33),
            (31, 17),
            (31, 24),
            (31, 29),
            (31, 33),
            (32, 33),
            (33, 0),
            (33, 17),
            (33, 24),
            (33, 28),
            (33, 29),
            (33, 33),
            (33, 34),
            (34, 33),
            (35, 17),
            (35, 24),
            (35, 29),
            (35, 33),
            (36, 33),
            (37, 0),
            (37, 17),
            (37, 24),
            (37, 28),
            (37, 29),
            (37, 33),
            (38, 33),
            (39, 0),
            (39, 17),
            (39, 24),
            (39, 28),
            (39, 29),
            (39, 33),
            (39, 34),
        }:
            return 5
        elif key in {(0, 34), (6, 34), (14, 34), (29, 34), (37, 34)}:
            return 15
        elif key in {(1, 34), (7, 34), (27, 34), (30, 34), (38, 34)}:
            return 25
        return 37

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_3_outputs, num_mlp_0_1_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_1_output, num_attn_1_3_output):
        key = (num_attn_1_1_output, num_attn_1_3_output)
        return 10

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_3_output, num_attn_0_1_output):
        key = (num_attn_1_3_output, num_attn_0_1_output)
        return 3

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_1_output):
        key = num_attn_1_1_output
        return 2

    num_mlp_1_2_outputs = [num_mlp_1_2(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_2_output, num_attn_1_0_output):
        key = (num_attn_0_2_output, num_attn_1_0_output)
        return 30

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"{", "("}:
            return position == 1
        elif token in {")", "}"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 6

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"("}:
            return position == 4
        elif token in {")", "}"}:
            return position == 5
        elif token in {"<s>", "{"}:
            return position == 6

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, position):
        if token in {"("}:
            return position == 5
        elif token in {")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 6
        elif token in {"{", "}"}:
            return position == 4

    attn_2_2_pattern = select_closest(positions, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, position):
        if token in {"("}:
            return position == 2
        elif token in {")", "}"}:
            return position == 5
        elif token in {"<s>"}:
            return position == 6
        elif token in {"{"}:
            return position == 1

    attn_2_3_pattern = select_closest(positions, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_2_output, mlp_0_0_output):
        if mlp_0_2_output in {0, 35, 4, 11, 30}:
            return mlp_0_0_output == 35
        elif mlp_0_2_output in {1, 2, 3, 9, 12, 17}:
            return mlp_0_0_output == 7
        elif mlp_0_2_output in {5}:
            return mlp_0_0_output == 4
        elif mlp_0_2_output in {6}:
            return mlp_0_0_output == 10
        elif mlp_0_2_output in {7}:
            return mlp_0_0_output == 28
        elif mlp_0_2_output in {8}:
            return mlp_0_0_output == 30
        elif mlp_0_2_output in {33, 10, 19, 39}:
            return mlp_0_0_output == 1
        elif mlp_0_2_output in {32, 13, 22, 26, 27, 29}:
            return mlp_0_0_output == 17
        elif mlp_0_2_output in {38, 14, 16, 20, 21, 23, 24, 31}:
            return mlp_0_0_output == 2
        elif mlp_0_2_output in {15}:
            return mlp_0_0_output == 14
        elif mlp_0_2_output in {18}:
            return mlp_0_0_output == 22
        elif mlp_0_2_output in {25, 28, 37, 36}:
            return mlp_0_0_output == 33
        elif mlp_0_2_output in {34}:
            return mlp_0_0_output == 32

    num_attn_2_0_pattern = select(mlp_0_0_outputs, mlp_0_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_0_1_output, attn_1_0_output):
        if num_mlp_0_1_output in {
            0,
            2,
            3,
            6,
            7,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            23,
            24,
            28,
            29,
            31,
            32,
            35,
            36,
        }:
            return attn_1_0_output == 7
        elif num_mlp_0_1_output in {
            1,
            5,
            8,
            10,
            17,
            18,
            19,
            20,
            21,
            22,
            25,
            26,
            27,
            30,
            33,
            34,
            37,
            38,
            39,
        }:
            return attn_1_0_output == 4
        elif num_mlp_0_1_output in {4}:
            return attn_1_0_output == 12

    num_attn_2_1_pattern = select(
        attn_1_0_outputs, num_mlp_0_1_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_2_output, attn_1_1_output):
        if attn_1_2_output in {0, 34, 36, 5, 38, 10, 15, 18}:
            return attn_1_1_output == "}"
        elif attn_1_2_output in {1, 6, 8, 9, 13, 19, 22, 24, 31}:
            return attn_1_1_output == ""
        elif attn_1_2_output in {
            2,
            3,
            4,
            7,
            11,
            12,
            14,
            16,
            17,
            20,
            21,
            23,
            25,
            26,
            27,
            28,
            29,
            30,
            32,
            33,
            35,
            37,
            39,
        }:
            return attn_1_1_output == ")"

    num_attn_2_2_pattern = select(attn_1_1_outputs, attn_1_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_2_output, mlp_0_0_output):
        if mlp_0_2_output in {
            0,
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
        }:
            return mlp_0_0_output == 2
        elif mlp_0_2_output in {1}:
            return mlp_0_0_output == 5

    num_attn_2_3_pattern = select(mlp_0_0_outputs, mlp_0_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_1_2_output, num_mlp_1_3_output):
        key = (mlp_1_2_output, num_mlp_1_3_output)
        if key in {
            (6, 1),
            (10, 1),
            (18, 0),
            (18, 1),
            (18, 4),
            (18, 7),
            (18, 16),
            (18, 30),
            (18, 38),
            (19, 1),
            (23, 0),
            (23, 1),
            (23, 4),
            (23, 7),
            (23, 15),
            (23, 16),
            (23, 17),
            (23, 18),
            (23, 19),
            (23, 21),
            (23, 24),
            (23, 25),
            (23, 26),
            (23, 27),
            (23, 30),
            (23, 38),
            (24, 1),
            (27, 1),
            (28, 1),
            (28, 4),
            (28, 16),
            (28, 30),
            (32, 1),
            (32, 4),
            (32, 30),
            (34, 0),
            (34, 1),
            (34, 4),
            (34, 7),
            (34, 15),
            (34, 16),
            (34, 19),
            (34, 21),
            (34, 24),
            (34, 25),
            (34, 27),
            (34, 30),
            (34, 38),
            (38, 0),
            (38, 1),
            (38, 4),
            (38, 7),
            (38, 15),
            (38, 16),
            (38, 19),
            (38, 24),
            (38, 25),
            (38, 27),
            (38, 30),
            (38, 38),
        }:
            return 5
        return 25

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_1_2_outputs, num_mlp_1_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_1_0_output, attn_1_3_output):
        key = (mlp_1_0_output, attn_1_3_output)
        if key in {
            (0, 5),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 5),
            (5, 18),
            (5, 30),
            (5, 31),
            (6, 5),
            (7, 5),
            (8, 5),
            (9, 5),
            (10, 5),
            (11, 5),
            (12, 5),
            (13, 5),
            (14, 5),
            (15, 5),
            (16, 5),
            (17, 5),
            (18, 5),
            (19, 5),
            (20, 5),
            (21, 5),
            (22, 5),
            (22, 30),
            (22, 31),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (23, 10),
            (23, 11),
            (23, 12),
            (23, 13),
            (23, 14),
            (23, 15),
            (23, 16),
            (23, 17),
            (23, 18),
            (23, 19),
            (23, 20),
            (23, 21),
            (23, 22),
            (23, 23),
            (23, 24),
            (23, 25),
            (23, 26),
            (23, 27),
            (23, 28),
            (23, 29),
            (23, 30),
            (23, 31),
            (23, 32),
            (23, 33),
            (23, 34),
            (23, 35),
            (23, 36),
            (23, 37),
            (23, 38),
            (23, 39),
            (24, 5),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 5),
            (25, 6),
            (25, 8),
            (25, 9),
            (25, 10),
            (25, 11),
            (25, 12),
            (25, 13),
            (25, 14),
            (25, 15),
            (25, 17),
            (25, 18),
            (25, 19),
            (25, 20),
            (25, 21),
            (25, 22),
            (25, 23),
            (25, 24),
            (25, 26),
            (25, 27),
            (25, 28),
            (25, 29),
            (25, 30),
            (25, 31),
            (25, 32),
            (25, 34),
            (25, 35),
            (25, 36),
            (25, 37),
            (25, 38),
            (25, 39),
            (26, 5),
            (27, 5),
            (28, 5),
            (29, 5),
            (30, 5),
            (31, 5),
            (32, 5),
            (33, 5),
            (34, 5),
            (35, 5),
            (36, 5),
            (37, 5),
            (38, 5),
            (38, 30),
            (38, 31),
            (39, 5),
        }:
            return 22
        return 11

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_1_0_outputs, attn_1_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_0_1_output, attn_1_3_output):
        key = (attn_0_1_output, attn_1_3_output)
        if key in {("(", 4), ("{", 4)}:
            return 34
        return 36

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_1_3_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_0_3_output, attn_2_3_output):
        key = (attn_0_3_output, attn_2_3_output)
        if key in {
            ("(", 0),
            ("(", 1),
            ("(", 2),
            ("(", 3),
            ("(", 4),
            ("(", 6),
            ("(", 7),
            ("(", 8),
            ("(", 9),
            ("(", 10),
            ("(", 11),
            ("(", 13),
            ("(", 14),
            ("(", 16),
            ("(", 17),
            ("(", 18),
            ("(", 19),
            ("(", 20),
            ("(", 21),
            ("(", 22),
            ("(", 23),
            ("(", 24),
            ("(", 25),
            ("(", 26),
            ("(", 27),
            ("(", 28),
            ("(", 29),
            ("(", 30),
            ("(", 31),
            ("(", 33),
            ("(", 35),
            ("(", 36),
            ("(", 37),
            ("(", 38),
            ("(", 39),
            ("{", 0),
            ("{", 1),
            ("{", 3),
            ("{", 4),
            ("{", 6),
            ("{", 7),
            ("{", 8),
            ("{", 9),
            ("{", 10),
            ("{", 11),
            ("{", 13),
            ("{", 14),
            ("{", 16),
            ("{", 17),
            ("{", 18),
            ("{", 19),
            ("{", 20),
            ("{", 21),
            ("{", 22),
            ("{", 23),
            ("{", 24),
            ("{", 25),
            ("{", 26),
            ("{", 27),
            ("{", 28),
            ("{", 29),
            ("{", 30),
            ("{", 31),
            ("{", 33),
            ("{", 35),
            ("{", 36),
            ("{", 37),
            ("{", 38),
            ("{", 39),
        }:
            return 32
        return 21

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_2_3_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_2_3_output):
        key = (num_attn_2_1_output, num_attn_2_3_output)
        return 33

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output, num_attn_2_1_output):
        key = (num_attn_1_2_output, num_attn_2_1_output)
        return 18

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_3_output, num_attn_2_2_output):
        key = (num_attn_2_3_output, num_attn_2_2_output)
        if key in {(0, 0), (0, 1)}:
            return 14
        return 36

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_0_output):
        key = num_attn_2_0_output
        return 16

    num_mlp_2_3_outputs = [num_mlp_2_3(k0) for k0 in num_attn_2_0_outputs]
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
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
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
            "}",
            ")",
            "(",
            "{",
            "}",
            "{",
            "}",
            "(",
            "}",
            "{",
            ")",
            "}",
            "}",
            ")",
            "}",
            "}",
            "}",
            "{",
            "(",
        ]
    )
)
