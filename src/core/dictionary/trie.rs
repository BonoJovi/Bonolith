use std::collections::HashMap;

pub struct TrieNode {
    children: HashMap<char, TrieNode>,
    /// Entry indices into Dictionary.entries, in insertion order. Callers that
    /// want frequency ordering (e.g. Dictionary::lookup) sort at read time.
    pub entry_indices: Vec<usize>,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            entry_indices: Vec::new(),
        }
    }
}

pub struct Trie {
    root: TrieNode,
}

impl Trie {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    /// Insert an entry index at the given reading. Insertion order is preserved
    /// on the node; callers that need frequency ordering sort at read time.
    pub fn insert(&mut self, reading: &str, entry_idx: usize) {
        let mut node = &mut self.root;
        for ch in reading.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        // Avoid duplicates
        if node.entry_indices.iter().any(|&idx| idx == entry_idx) {
            return;
        }
        node.entry_indices.push(entry_idx);
    }

    /// Exact lookup: return entry indices for the exact reading.
    pub fn exact_lookup(&self, reading: &str) -> &[usize] {
        let mut node = &self.root;
        for ch in reading.chars() {
            match node.children.get(&ch) {
                Some(child) => node = child,
                None => return &[],
            }
        }
        &node.entry_indices
    }

    /// Common prefix search: find all prefixes of `input` that exist as dictionary entries.
    /// Returns Vec of (prefix_char_length, entry_indices).
    pub fn common_prefix_search(&self, input: &str) -> Vec<(usize, Vec<usize>)> {
        let mut results = Vec::new();
        let mut node = &self.root;
        let mut char_len = 0;

        for ch in input.chars() {
            match node.children.get(&ch) {
                Some(child) => {
                    node = child;
                    char_len += 1;
                    if !node.entry_indices.is_empty() {
                        results.push((char_len, node.entry_indices.clone()));
                    }
                }
                None => break,
            }
        }
        results
    }

    /// Prefix lookup: return all entry indices under the given prefix.
    pub fn prefix_lookup(&self, prefix: &str) -> Vec<usize> {
        let mut node = &self.root;
        for ch in prefix.chars() {
            match node.children.get(&ch) {
                Some(child) => node = child,
                None => return Vec::new(),
            }
        }
        let mut results = Vec::new();
        Self::collect_all(node, &mut results);
        results
    }

    /// Remove an entry index from the node for the given reading.
    pub fn remove(&mut self, reading: &str, entry_idx: usize) {
        let mut node = &mut self.root;
        for ch in reading.chars() {
            match node.children.get_mut(&ch) {
                Some(child) => node = child,
                None => return,
            }
        }
        node.entry_indices.retain(|&idx| idx != entry_idx);
    }

    fn collect_all(node: &TrieNode, results: &mut Vec<usize>) {
        results.extend_from_slice(&node.entry_indices);
        for child in node.children.values() {
            Self::collect_all(child, results);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_exact_lookup() {
        let mut trie = Trie::new();
        trie.insert("きょう", 0);
        trie.insert("きょう", 1);
        trie.insert("きた", 2);

        assert_eq!(trie.exact_lookup("きょう"), &[0, 1]);
        assert_eq!(trie.exact_lookup("きた"), &[2]);
    }

    #[test]
    fn exact_lookup_miss() {
        let trie = Trie::new();
        assert_eq!(trie.exact_lookup("きょう"), &[] as &[usize]);
    }

    #[test]
    fn common_prefix_search_basic() {
        let mut trie = Trie::new();
        trie.insert("き", 0);
        trie.insert("きょう", 1);
        trie.insert("きょうと", 2);

        let results = trie.common_prefix_search("きょうは");
        // Should find: き (len=1), きょう (len=3), but NOT きょうと (input too short at は)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, vec![0]));     // き
        assert_eq!(results[1], (3, vec![1]));     // きょう
    }

    #[test]
    fn prefix_lookup_basic() {
        let mut trie = Trie::new();
        trie.insert("きょう", 0);
        trie.insert("きょうと", 1);
        trie.insert("きょうだい", 2);
        trie.insert("きた", 3);

        let mut results = trie.prefix_lookup("きょう");
        results.sort();
        assert_eq!(results, vec![0, 1, 2]);

        let mut results2 = trie.prefix_lookup("き");
        results2.sort();
        assert_eq!(results2, vec![0, 1, 2, 3]);
    }

    #[test]
    fn insert_deduplicates_repeat_entry_idx() {
        // Regression: previously used partition_point with a non-monotonic
        // predicate, which returns an unspecified index and could let the
        // same entry_idx be pushed twice when it sits mid-vector.
        let mut trie = Trie::new();
        trie.insert("あ", 10);
        trie.insert("あ", 20);
        trie.insert("あ", 30);
        trie.insert("あ", 40);
        trie.insert("あ", 50);
        // Re-insert a middle idx — must not duplicate.
        trie.insert("あ", 30);
        assert_eq!(trie.exact_lookup("あ"), &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn empty_trie() {
        let trie = Trie::new();
        assert!(trie.exact_lookup("あ").is_empty());
        assert!(trie.prefix_lookup("あ").is_empty());
        assert!(trie.common_prefix_search("あいう").is_empty());
    }
}
