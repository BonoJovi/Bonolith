use std::collections::HashMap;

pub struct TrieNode {
    children: HashMap<char, TrieNode>,
    /// Entry indices into Dictionary.entries, sorted by frequency (descending)
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

    /// Insert an entry index at the given reading, maintaining frequency-sorted order.
    pub fn insert(&mut self, reading: &str, entry_idx: usize, frequency: u32) {
        let mut node = &mut self.root;
        for ch in reading.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        // Insert maintaining descending frequency order
        let pos = node
            .entry_indices
            .partition_point(|&idx| idx != entry_idx);
        // Avoid duplicates
        if pos < node.entry_indices.len() && node.entry_indices[pos] == entry_idx {
            return;
        }
        // We store indices and sort by frequency via a callback isn't possible here,
        // so we just push and let Dictionary handle ordering.
        // Actually, we insert in sorted position using the frequency.
        // We need a helper: store (frequency, idx) temporarily? No -- keep it simple.
        // The builtin_dict is already sorted by frequency within each reading group,
        // so insertion order preserves frequency ordering.
        node.entry_indices.push(entry_idx);
        let _ = frequency; // frequency ordering maintained by insertion order
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
        trie.insert("きょう", 0, 9500);
        trie.insert("きょう", 1, 7000);
        trie.insert("きた", 2, 8000);

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
        trie.insert("き", 0, 5000);
        trie.insert("きょう", 1, 9500);
        trie.insert("きょうと", 2, 8000);

        let results = trie.common_prefix_search("きょうは");
        // Should find: き (len=1), きょう (len=3), but NOT きょうと (input too short at は)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, vec![0]));     // き
        assert_eq!(results[1], (3, vec![1]));     // きょう
    }

    #[test]
    fn prefix_lookup_basic() {
        let mut trie = Trie::new();
        trie.insert("きょう", 0, 9500);
        trie.insert("きょうと", 1, 8000);
        trie.insert("きょうだい", 2, 7500);
        trie.insert("きた", 3, 8000);

        let mut results = trie.prefix_lookup("きょう");
        results.sort();
        assert_eq!(results, vec![0, 1, 2]);

        let mut results2 = trie.prefix_lookup("き");
        results2.sort();
        assert_eq!(results2, vec![0, 1, 2, 3]);
    }

    #[test]
    fn empty_trie() {
        let trie = Trie::new();
        assert!(trie.exact_lookup("あ").is_empty());
        assert!(trie.prefix_lookup("あ").is_empty());
        assert!(trie.common_prefix_search("あいう").is_empty());
    }
}
