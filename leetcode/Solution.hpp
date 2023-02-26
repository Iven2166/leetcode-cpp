//
//  Solution.hpp
//  leetcode
//
//  Created by tmp on 2022/12/25.
//

#ifndef Solution_hpp
#define Solution_hpp

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <queue>
using namespace std;

struct ListNode{
    int val;
    ListNode* next;
    ListNode(): val(0), next(nullptr) {}
    ListNode(int x): val(x), next(nullptr) {}
    ListNode(int x, ListNode* next): val(x), next(next) {}
};

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(): val(0), left(nullptr), right(nullptr){}
    TreeNode(int x): val(x), left(nullptr), right(nullptr){}
    TreeNode(int x, TreeNode* left, TreeNode* right): val(x), left(left), right(right){}
};

class Solution {
public:
    
    // 112.路径总和
    bool hasPathSumBFS(TreeNode* root, int targetSum){
        // 使用队列去进行bfs算法
        if(root == nullptr)
            return false;
        queue<TreeNode*> q_tree;
        queue<int> q_val;
        q_tree.push(root);
        q_val.push(root->val);
        while(!q_tree.empty()){
            // 队列非空时
            // 取队列头部
            TreeNode* curr = q_tree.front();
            int curr_val = q_val.front();
            // 弹出
            q_tree.pop();
            q_val.pop();
            
            if(curr->left == nullptr && curr->right == nullptr){
                if(curr_val == targetSum)
                    return true;
                continue;
            }
            // 填入队列，基于父子树已累计的val，添加本节点val的的操作
            if(curr->left != nullptr){
                q_tree.push(curr->left);
                q_val.push(curr_val + curr->left->val);
            }
            if(curr->right != nullptr){
                q_tree.push(curr->right);
                q_val.push(curr_val + curr->right->val);
            }
        }
        return false;
    }
    
    bool hasPathSum(TreeNode* root, int targetSum){
        bool ans = false;
        hasPathSumCore(root, 0, targetSum, ans);
        return ans;
    }
    
    void hasPathSumCore(TreeNode* cur, int cur_sum, int targetSum, bool& ans){
        if(cur->left){
            hasPathSumCore(cur->left, cur_sum + cur->val, targetSum, ans);
        }
        if(cur->right){
            hasPathSumCore(cur->left, cur_sum + cur->val, targetSum, ans);
        }
        if(cur->left==nullptr && cur->right==nullptr){
            if(cur_sum + cur->val == targetSum){
                ans |= true;
            }
            return;
        }
    }
    
    //301.删除无效的括号
    vector<string> removeInvalidParentheses1(string s) {
        vector<string> ans;
        // 统计无效括号个数，然后DFS生成
        int n = s.size();
        int left_cnt = 0, invalid_cnt = 0;
        int i = 0;
        while(i < n){
            if(s[i] == '('){
                left_cnt++;
            }
            else{
                if(left_cnt > 0){
                    left_cnt--;
                }
                else{
                    invalid_cnt++;
                }
            }
            i++;
        }
        invalid_cnt += left_cnt;
        removeInvalidParentheses1Dfs(ans, "", s, left_cnt, 0, invalid_cnt, n);
        return ans;
    }
    
    void removeInvalidParentheses1Dfs(vector<string>& ans, string curr, string s,
                                      int left_cnt, int pos, int invalid_cnt, int n){
        /* curr 是目前所拼接的string括号串，left_cnt 是 curr 里左括号数量，pos是目前遍历到的括号位置
         */
        if(pos == n){
            if(left_cnt == 0 && curr.size() == n - invalid_cnt){
                ans.push_back(curr);
            }
            return ;
        }
        else if(curr.size() > n - invalid_cnt || curr.size() + left_cnt > n - invalid_cnt){
            return ;
        }
        else {
            // 跳过的情况
            removeInvalidParentheses1Dfs(ans, curr, s, left_cnt, pos + 1, invalid_cnt, n);
            // 左括号情况
            if(s[pos] == '('){
                curr.push_back('(');
                removeInvalidParentheses1Dfs(ans, curr, s, left_cnt + 1, pos + 1, invalid_cnt, n);
                curr.pop_back();
            }
            // 右括号情况
            else if(s[pos] == ')'){
                if(left_cnt <= 0)
                    return ;
                curr.push_back(')');
                removeInvalidParentheses1Dfs(ans, curr, s, left_cnt - 1, pos + 1, invalid_cnt, n);
                curr.pop_back();
            }
            // 普通字符情况
            else{
                curr.push_back(s[pos]);
                removeInvalidParentheses1Dfs(ans, curr, s, left_cnt, pos + 1, invalid_cnt, n);
                curr.pop_back();
            }
        }
        
    }
    
    
    //32. 最长有效括号
    int longestValidParentheses(string s) {
        int n = s.size();
        int ans = 0;
        vector<vector<bool>> dp;
        dp.resize(n);
        for(int i=0; i<n; i++)
        {
            dp[i].resize(n);
        }
        
        for(int i=0; i<n-1; i++){
            dp[i][i+1] = dp[i][i+1] | ((s[i]=='(' && s[i+1]==')') ? true : false);
        }
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                if(i+1>0 && j-1<n && i+1<j-1){
                    if(s[i]=='(' && s[j]==')')
                        dp[i][j] = dp[i][j] | dp[i+1][j-1];
                }
                if(i+2>0 && j<n && i+2<j){
                    if(s[i]=='(' && s[i+1]==')')
                        dp[i][j] = dp[i][j] | dp[i+2][j];
                }
                if(i>0 && j-2<n && i+2<j){
                    if(s[j-1]=='(' && s[j]==')')
                        dp[i][j] = dp[i][j] | dp[i][j-2];
                }
                if(dp[i][j] && j-i+1 > ans)
                    ans = j-i+1;
            }
        }
        return ans;
    }
    
    vector<string> generateParenthesis2(int n) {
        vector<string> ans;
        unordered_map<char, char> hashmap;
        hashmap['('] = ')';
        string cur = "";
        generateParenthesisCore2(ans, cur, n, 0);
        return ans;
    }
    void generateParenthesisCore2(vector<string>& ans, string cur, int n, int left_cnt){
        if(cur.size() == n * 2){
            if (left_cnt == 0)
                ans.push_back(cur);
            return;
        }
        else if(cur.size() > n * 2){
            return;
        }
        else{
            if(left_cnt == 0){
                cur.push_back('(');
                generateParenthesisCore2(ans, cur, n, left_cnt + 1);
            }
            else{
                cur.push_back('(');
                generateParenthesisCore2(ans, cur, n, left_cnt + 1);
                cur.pop_back();
                cur.push_back(')');
                generateParenthesisCore2(ans, cur, n, left_cnt - 1);
                cur.pop_back();
            }
        }
        
    }
    
    // 16. 最接近的三数之和
    
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int ans = 0;
        int gap = INT_MAX;
        for(int i = 0; i < nums.size() - 2; i++){
            int t = nums[i];
            int left = i + 1, right = nums.size() - 1;
            while(left < right){
                if(left < right && abs(t + nums[left] + nums[right] - target)< gap){
                    ans = t + nums[left] + nums[right];
                    // cout << nums[i] << ',' << nums[left] << ',' << nums[right] << endl;
                    gap = abs(target - ans);
                }
                if(nums[left] + nums[right] + t == target){
                    return target;
                }
                else if(nums[left] + nums[right] + t < target){
                    left++;
                }
                else {
                    right--;
                }
                
            }
        }
        return ans;
    }
    //541. 反转字符串 II
    string reverseStr(string s, int k) {
        int n = s.size();
        for(int i=0; i < n; i += 2*k){
            reverse(s.begin() + i, s.begin() + min(i + k, n));
        }
        return s;
    }
    // 557. 反转字符串中的单词 III
    string reverseWords(string s) {
        string ans;
        int i = 0, length = s.size();
        while(i < length){
            int start = i;
            while(i < length && s[i] != ' '){
                i++;
            }
            for(int j=start; j<i; j++){
                ans.push_back(s[i + start - j - 1]);
            }
            while(i < length && s[i]==' '){
                i++;
                ans.push_back(' ');
            }
        }
        return ans;
    }
    
    // 95. 不同的二叉搜索树 II
    vector<TreeNode*> generateTrees(int n) {
        return generateTreesCore(1, n);
    }
    vector<TreeNode*> generateTreesCore(int left, int right){
        if(right < left){
            return {nullptr};
        }
        vector<TreeNode*> list_treenode;
        for(int mid = left; mid <= right; mid++){
            vector<TreeNode*> left_trees = generateTreesCore(left, mid-1);
            vector<TreeNode*> right_trees = generateTreesCore(mid+1, right);
            for(auto left_tr: left_trees){
                for(auto right_tr: right_trees){
                    TreeNode * mid_tr = new TreeNode(mid);
                    mid_tr->left = left_tr;
                    mid_tr->right = right_tr;
                    list_treenode.push_back(mid_tr);
                }
            }
        }
        return list_treenode;
    }
    
    // 264. 丑数 II
    int nthUglyNumber(int n) {
        // 开辟 2 * n 长度的数组，存储每个数是否为丑数的状态，初始化为 0
        if(n==1){return 1;}
        priority_queue<long, vector<long>, greater<long> > ques; // 小根堆
        unordered_set<long> seen;
        ques.push(1L);
        seen.insert(1L);
        int ans = 0;
        for(int i=0; i<n; i++){
            ans = ques.top();
            ques.pop();
            int ugly = (int) ans;
            for(int factor: {2,3,5}){
                long next = (long) factor * ugly;
                if(!seen.count(next)){
                    seen.insert( next);
                    ques.push(next);
                }
            }
        }
        return ans;
    }
    
    // 204
    int countPrimes(int n) {
        if(n==0){return 0;}
        vector<int> isPrime(n, 1);
        int cnt = 0;
        // notPrime[1] = 1;
        for(int i=2; i<n; i++){
            if(isPrime[i]==1){
                cnt++;
            }
            if((long long)i*i < n){
                for(int j=i*i; j<n; j+=i){
                    isPrime[j] = 0;
                }
            }
            
        }
        return cnt;
    }
    
    // 287.
    int findDuplicate(vector<int>& nums) {
        //1. 诸位寻找
        // int ans=0;
        // for(int i=0; i<32;i++){
        //     int p1=0,p2=0;
        //     for(int j=0; j<nums.size(); j++){
        //         if(((nums[j]>>i) & 1) == 1){p1++;}
        //         if(j>0 && ((j>>i) & 1)){p2++;}
        //     }
        //     if(p2<p1){ans += (1<<i);}
        // }
        // return ans;
        // 2.环形链表
        // int slow = 0, fast = 0;
        // do{
        //     slow = nums[slow];
        //     fast = nums[nums[fast]];
        // }while(slow!=fast);
        // slow = 0;
        // do{
        //     slow = nums[slow];
        //     fast = nums[fast];
        // }while(slow!=fast);
        // return slow;
        // 3.二分查找
        int left = 1, right = nums.size() - 1, ans=-1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            int cnt = 0;
            for(int i = 0; i<nums.size(); i++){
                if(nums[i] <= mid){cnt++;}
            }
            if(cnt <= mid){left = mid + 1;}
            else{right = mid - 1; ans = mid;}
        }
        return ans;
    }
    
    // 树的中序遍历
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> stk;
        while(root!=nullptr || !stk.empty()){
            // 当前root就是该访问的root，先判断左树，放到stk里，作为访问树的顺序
            while(root!=nullptr){
                stk.push(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            res.push_back(root->val);
            root = root->right;
        }
        return res;
    }
    
    //面试题 17.09. 第 k 个数
    int getKthMagicNumber(int k) {
        // vector<int> stk;
        // for(int m=0;m<k;m++){
        //     for(int j=0;j<k;j++){
        //         for(int i=0; i<k; i++){
        //             stk.push_back(pow(3, i) * pow(5, j) * pow(7, m));
        //         }
        //     }
        // }
        // sort(stk.begin(), stk.end());
        // for(auto& iter: stk){
        //     cout<< iter<<',';
        // }
        // return 0;
        int p3=0, p5=0, p7=0;
        vector<int> result;
        result.push_back(1);
        for(int i=0; i<k; i++){
            int res = min(result[p3]*3, min(result[p5]*5, result[p7]*7));
            if(res==result[p3]*3){
                p3++;
            }
            if(res==result[p5]*5){
                p5++;
            }
            if(res==result[p7]*7){
                p7++;
            }
            result.push_back(res);
        }
        return result[k-1];
    }
    
    // 322周赛
    
    int minScore(int n, vector<vector<int>>& roads) {
        vector<vector<pair<int,int>>> dict(n+1);
        for(auto& iter: roads){
            dict[iter[0]].emplace_back(iter[1],iter[2]);
            dict[iter[1]].emplace_back(iter[0],iter[2]);
        }
        int ans = INT_MAX;
        queue<int> q;
        vector<int> visited(n+1);
        q.push(1);
        visited[1] = 1;
        while(not q.empty()){
            int cur = q.front();
            q.pop();
            for(auto [v, w]: dict[cur]){
                ans = min(w, ans);
                if((not (visited[v]==1))){
                    q.push(v);
                    visited[v]=1;
                }
            }
        }
        return ans;
    }
    
    long long dividePlayers(vector<int>& skill) {
        int n = skill.size();
        int k = n / 2; // k个组
        int sum = 0;
        for(auto& iter: skill)
            sum += iter;
        if(sum % k == 1){
            return -1;
        }
        int target = sum / k;
        long long ans = 0;
        
        sort(skill.begin(), skill.end());
        for(int i=0; i<n/2; i++){
            if(skill[i] + skill[n-1-i] == target){
                ans += skill[i] * skill[n-1-i];
            }
            else{
                return -1;
            }
        }
        
        return -1;
    }
    
    // 698. 划分为k个相等的子集
    bool canPartitionKSubsets(vector<int>& nums, int k){
        // 分到k个桶里，做回溯
        int* bucket = new int[k];
        for(int i=0; i<k;i++){
            bucket[i] = 0;
        }
        int sum = 0;
        for(auto& iter: nums)
            sum += iter;
        if(sum%k)
            return false;
        
        int target = sum / k;
        sort(nums.begin(), nums.end());
        int left=0, right=nums.size()-1;
        while(left <= right){
            int tmp = nums[right];
            nums[right] = nums[left];
            nums[left] = tmp;
            left++;
            right--;
        }
        if(nums[0] > target){
            return false;
        }
        bool ans = canPartitionKSubsetsCore(nums, 0, bucket, k, target);
        return ans;
    }
    bool canPartitionKSubsetsCore(vector<int>& nums, int index, int bucket[], int k, int target){
        if(index == nums.size())
            return true;// 处理完所有的球
        for(int i=0; i<k; i++){
            // 当前这个球，选择k个箱
            if(i>0 && bucket[i]==bucket[i-1])
                continue; // 说明这个球加这俩个箱效果一样，可以跳过
            if(nums[index] + bucket[i] > target)
                continue;
            bucket[i] += nums[index];
            if(canPartitionKSubsetsCore(nums, index + 1, bucket, k, target))
                return true;
            bucket[i] -= nums[index];
        }
        return false;
    }
    
    
    //416. 分割等和子集
    bool canPartition(vector<int>& nums) {
        // 判断特殊false情况
        int sum = 0;
        int max = 0;
        for(auto& iter:nums){
            sum += iter;
            max = (iter > max) ? iter : max;
        }
        
        if(sum%2==1 || nums.size()==1)
            return false;
        int target = sum / 2; // 排除不能整除情况，找到一半即可，剩余一半自动合理。
        if(max > target)
            return false;
        // dp[i][j] 代表由nums[0:i+1] 里选择若干数字，能够构成 j 总和
        vector<vector<bool>> dp(nums.size(), vector<bool>(target+1, false));
        // bool dp[nums.size()][target+1] ;
        // 初始化边界: dp[i][0]=true 什么都不选择，为0 , dp[0][nums[0]]=true 选择
        for(int i=0;i<nums.size(); i++)
            dp[i][0] = true;
        dp[0][nums[0]] = true;
        for(int i=1; i<nums.size(); i++){
            for(int j=1; j<=target; j++){
                dp[i][j] = (dp[i-1][j] | dp[i][j]);
                if(j >= nums[i])
                    dp[i][j] = (dp[i-1][j-nums[i]] | dp[i][j]);
            }
        }
        return dp[nums.size()-1][target];
    }
    
    
    // 215. 数组中的第K个最大元素
    int findKthLargest(vector<int>& nums, int k) {
        // 降序排列
        int left=0, right=nums.size()-1;
        while(true){
            int pivot = findKthLargestCore(nums, left, right);
            if(pivot == k - 1){
                return nums[pivot];
            }
            else if(pivot < k - 1){
                left = pivot + 1;
            }
            else{
                right = pivot - 1;
            }
        }
    }
    
    int findKthLargestCore(vector<int>& nums, int left, int right){
        int pivot = nums[left];
        while(left < right){
            while(left < right && nums[right] <= pivot){
                right--;
            }
            nums[left] = nums[right];
            while(left < right && nums[left] >= pivot){
                left++;
            }
            nums[right] = nums[left];
        }
        nums[left] = pivot;
        return left;
    }
    
    //53. 最大子数组和
    int maxSubArray(vector<int>& nums) {
        if(nums.size()==1)
            return nums[0];
        int dp = nums[0], ans = nums[0];
        for(int i=1; i<nums.size(); i++){
            if(dp<0){
                dp = nums[i];
            }
            else{
                dp += nums[i];
            }
            ans = (ans < dp) ? dp : ans;
        }
        return ans;
    }
    
    
    // 21. 合并两个有序链表
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* dummy = new ListNode(-101, new ListNode);
        ListNode* p1 = list1;
        ListNode* p2 = list2;
        ListNode* node = dummy;
        //        ListNode* tmp;
        while(p1 || p2){
            if(p1==nullptr){
                node->next = p2;
                break;
            }
            if(p2==nullptr){
                node->next = p1;
                break;
            }
            // p1 and p2
            if(p1->val <= p2->val){
                ListNode* tmp = new ListNode(p1->val);
                p1 = p1->next;
                node->next = tmp;
            }
            else{
                ListNode* tmp = new ListNode(p2->val);
                p2 = p2->next;
                node->next = tmp;
            }
            node = node->next;
        }
        return dummy->next;
    }
    
    // 72. 编辑距离
    int minDistance(string word1, string word2) {
        /*
         给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
         dp[i][j] 表示以 word1[0:i] 改为 word2[0:j] 的次数
         初始化
         dp[0][j] = j (j for all) 即增加 j 个字符
         dp[i][0] = i (i for all) 即删除 i 个字符
         转移矩阵
         tmp = (word1[i-1]==word2[j-1]) ? dp[i-1][j-1] : len1+len2;
         dp[i][j] = min(tmp, dp[i-1][j]+1, dp[i][j-1]+1);
         */
        int ans=0;
        int len1 = word1.size(), len2 = word2.size();
        vector<vector<int>> dp(len1+1, vector<int>(len2+1));
        
        for(int i=0; i<len1+1; i++)
            dp[i][0] = i;
        for(int j=0; j<len2+1; j++)
            dp[0][j] = j;
        
        for(int i=1; i<len1+1; i++){
            for(int j=1; j<len2+1; j++){
                int m1 = min(dp[i-1][j]+1, dp[i][j-1]+1);
                int m2 = dp[i-1][j-1];
                m2 += (word1[i-1]==word2[j-1]) ? 0 : 1;
                dp[i][j] = min(m1, m2);
            }
        }
        return dp[len1][len2];
    }
    
    // 11. 盛最多水的容器
    int maxArea(vector<int>& height) {
        /*
         1,8,6,2,5,4,8,3,7
         
         */
        
        int left=0, right=height.size()-1;
        if(height.size()==2){
            return min(height[0], height[1]);
        }
        int ans = 0;
        while(left<=right){
            int w = right - left;
            int h = min(height[right], height[left]);
            ans = (w * h > ans) ? w * h : ans;
            if(height[left]<=height[right]){
                left++;
            }
            else{
                right--;
            }
            
        }
        return ans;
    }
    
    // 22. 括号生成
    vector<string> generateParenthesis(int n) {
        /* dfs 函数,有两个辅助变量。1、记录目前的左括号数 left_cnt 2、目前的字符串长度
         需要字符串长度 < n * 2
         当字符串为空或者 left_cnt==0，只能接 (, left_cnt++
         当 left_cnt > 0
         可以接 ( , left_cnt++
         可以接 ) , left_cnt--
         left_cnt > n: 非法，直接 return
         字符串长度==n*2
         检查 left_cnt == 0 才可以进 vector
         */
        string cur = "";
        int left_cnt = 0;
        vector<string> res;
        generateParenthesisCore(left_cnt, cur, n, res);
        return res;
    }
    void generateParenthesisCore(int left_cnt, string cur, int n, vector<string>& res){
        if(cur.size() < n*2){
            if(cur=="" || left_cnt == 0){
                cur.push_back('(');
                generateParenthesisCore(left_cnt+1, cur, n, res);
            }
            else if(left_cnt>0){
                cur.push_back('(');
                generateParenthesisCore(left_cnt+1, cur, n, res);
                cur.pop_back();
                cur.push_back(')');
                generateParenthesisCore(left_cnt-1, cur, n, res);
                cur.pop_back();
            }
        }
        else if(cur.size()== n*2){
            if(left_cnt==0)
                res.push_back(cur);
            return;
        }
    }
    
    // 200. 岛屿数量
    int numIslands(vector<vector<char>>& grid) {
        /*
         dfs往四个方向走，走到的地方变为 '0'
         */
        int row = grid.size();
        int col = grid[0].size();
        int ans = 0;
        for(int r=0; r<row; r++){
            for(int c=0; c<col; c++){
                if(grid[r][c]=='1'){
                    ans += 1;
                    numIslandsDFS(grid, r, c, row, col);
                }
            }
        }
        return ans;
    }
    void numIslandsDFS(vector<vector<char>>& grid, int r, int c, int row, int col){
        // 排除'0'情况，以及越界情况
        if((r > row - 1) || (r < 0) || (c < 0) || (c > col - 1))
            return;
        if(grid[r][c]=='0')
            return;
        
        // 此刻 grid[r][c]=='1'
        grid[r][c] = '0';
        vector<vector<int>> pairs = {{1,0},{-1,0},{0,1},{0,-1}};
        for(auto& iter: pairs){
            //            cout << iter[0] << ',' << iter[1] << endl;
            numIslandsDFS(grid, r+iter[0], c+iter[1], row, col);
        }
        return;
    }
    
    // 42. 接雨水
    int trap1(vector<int>& height){
        /* 单调栈：是“逐层”地进行雨水统计的
         栈是存储val单调递减的数值
         while 条件：当stack非空时 and val 大于 stack.top时，
         需要pop后(作为cur高度)，左边还有一个数值作为left
         width = i - left - 1
         height = min(height[left],height[i])-cur
         */
        int len = height.size();
        stack<int> stk;
        int ans = 0;
        for(int i=0; i<len; i++){
            while(!stk.empty() && height[i]>height[stk.top()]){
                // 栈非空并且当前大于top才需要处理
                int top_idx = stk.top();
                stk.pop();
                // 左侧还需要有值，否则退出
                if(stk.empty())
                    break;
                int left = stk.top();
                int cur_width = i - left - 1;
                int cur_height = min(height[i], height[left]) - height[top_idx];
                ans += cur_width * cur_height;
            }
            stk.push(i);
        }
        return ans;
    }
    
    int trap(vector<int>& height) {
        // 每个“水柱”由左右的最高所决定，所以需要存储每个位置对应的左最高高度，右最高高度。
        int n = height.size();
        vector<int> left_h(n, 0);
        vector<int> right_h(n, 0);
        int ans = 0;
        left_h[0] = 0;
        int left_max = 0, right_max = 0;
        for(int i=1; i<n; i++){
            left_max = (height[i-1]>left_max) ? height[i-1] : left_max;
            left_h[i] = left_max;
        }
        right_h[right_h.size()-1] = 0;
        for(int i=n-2; i>=0; i--){
            right_max = (height[i+1]>right_max) ? height[i+1] : right_max;
            right_h[i] = right_max;
        }
        for(int i=1; i<n-1; i++){
            int min = left_h[i]<right_h[i] ? left_h[i] : right_h[i];
            ans += (min <= height[i]) ? 0 : min - height[i];
        }
        return ans;
    }
    
    // 15. 三数之和
    vector<vector<int>> threeSum(vector<int>& nums) {
        /* 排序。三个数字加和为0，说明第一个数字为负数，
         再看第二个数字（需要跟上一个比较，如果相同则跳过），在此进行二分查找.
         */
        sort(nums.begin(), nums.end());
        int len = nums.size();
        vector<vector<int>> ans;
        //        int left, right;
        for(int i=0; i<len-2; i++){
            if((nums[i] > 0) | (nums[i]+nums[i+1]+nums[i+2] > 0)) // 剪枝
                return ans;
            if(i>=1 & nums[i]==nums[i-1]){
                continue;
            }
            int target = - nums[i];
            int right = len - 1;
            for(int left=i+1; left < len-1; left++){
                right = len - 1;
                if(left + 1 < right & nums[left]==nums[left+1]){
                    left ++ ;
                    continue;
                }
                while(left < right & nums[left] + nums[right] > target)
                    right--;
                if(nums[left] + nums[right] == target)
                    ans.push_back({nums[i], nums[left], nums[right]});
            }
        }
        return ans;
    }
    
    //20. 有效的括号
    bool isValid(string s) {
        unordered_map <char, char> hashmap = {{')','('},{'}','{'},{']','['}};
        //        vector<vector<char>> pairs = {{'(',')'},{'{','}'},{'[',']'}};
        //        for(auto& iter: pairs)
        //            hashmap[iter[1]] = iter[0];
        vector<char> stack;
        // 遍历s
        // 栈空：当前为左括号，则push，为右括号，直接返回false。
        // 栈非空：当前为左，则push。为右括号，查找栈顶是否为左括号，是则弹出；否则返回false。
        for(auto& cur: s){
            if(cur=='{'|cur=='('|cur=='['){
                stack.push_back(cur);
            }
            else{
                if(stack.size()<=0){
                    return false;
                }
                else{
                    if(stack.back()==hashmap[cur]){
                        stack.pop_back();
                    }
                    else{
                        return false;
                    }
                }
            }
        }
        return stack.empty() ? true: false;
    }
    
    // 3. 无重复字符的最长子串
    int lengthOfLongestSubstring(string s) {
        if(s=="")
            return 0;
        // 从左到右遍历。用一个hashmap保存当前字符出现的最新位置 pos（应该是最右边）。查找当下位置i对应字符的pos，如果大于start，说明start应该更新为 pos+1；反之不理。pos所在字符的map-value更新为i
        unordered_map<char, int> hashmap;
        int start = 0;
        hashmap[s[0]] = 0;
        int len_max = 1;
        for(int i=1; i<s.size(); i++){
            auto iter = hashmap.find(s[i]);
            if((iter!=hashmap.end())){
                if( (iter->second >= start)){
                    start = iter->second + 1;
                }
            }
            hashmap[s[i]] = i;
            if(i-start+1>len_max)
                len_max = i - start + 1;
        }
        return len_max;
    }
    
    
    // 2. 两数相加
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // 两个链表的逐个数字增加
        int up10 = 0; // 超过10，下一位增加1
        ListNode* res = new ListNode(-1);
        ListNode* node = res;
        while(l1 || l2){
            int n1 = l1 ? l1->val :0;
            int n2 = l2 ? l2->val :0;
            ListNode* cur = new ListNode((n1+n2+up10)%10);
            up10 = (n1+n2+up10>=10) ? 1:0;
            l1 = l1 ? l1->next :nullptr;
            l2 = l2 ? l2->next :nullptr;
            node->next = cur;
            node = node->next;
        }
        if(up10 > 0)
            node->next = new ListNode(1);
        return res->next;
    }
    
    // 4. 寻找两个正序数组的中位数
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        /*
         输入：nums1 = [1,3], nums2 = [2]
         输出：2.00000
         解释：合并数组 = [1,2,3] ，中位数 2
         */
        // k作为两个长度的中间位置
        int len1 = nums1.size();
        int len2 = nums2.size();
        
        if((len1 + len2)%2==1){
            return findMedianSortedArraysCore(nums1, nums2, len1, len2, (len1+len2)/2+1);
        }
        else{
            return (findMedianSortedArraysCore(nums1, nums2, len1, len2, (len1+len2)/2) + findMedianSortedArraysCore(nums1, nums2, len1, len2, (len1+len2)/2+1))/2.0;
        }
        return 0.0;
    }
    
    double findMedianSortedArraysCore(vector<int>& nums1, vector<int>& nums2, int len1, int len2, int k){
        int p1 = 0, p2 = 0; // 代表指向两个数组的指针
        while(true){
            if(p1==len1)
                return nums2[p2+k-1];
            else if(p2==len2)
                return nums1[p1+k-1];
            if(k==1)
                return min(nums1[p1], nums2[p2]);
            
            int next_p1 = min(p1 + k/2 - 1, len1 - 1);
            int next_p2 = min(p2 + k/2 - 1, len2 - 1);
            if(nums1[next_p1] < nums2[next_p2]){
                k -= (next_p1 - p1 + 1);
                p1 = next_p1 + 1;
            }
            else{
                k -= (next_p2 - p2 + 1);
                p2 = next_p2 + 1;
            }
        }
        return 0.0;
    }
    
    
    
    //    1. 两数之和
    vector<int> twoSum(vector<int>& nums, int target) {
        // 逻辑：存储hashmap，遍历时寻找。一次遍历即可，因为合为target的两个数是对称可找的。
        unordered_map<int, int> hashmap;
        for(int i=0; i<nums.size(); i++){
            auto iter = hashmap.find(target - nums[i]);
            if(iter != hashmap.end()){
                // 说明找到
                return {i, iter->second};
            }
            hashmap[nums[i]] = i;
        }
        return {-1,-1}; // 未找到的情况
    }
    //    5. 最长回文子串
    string longestPalindrome(string s) {
        long len = s.size();
        int plus = 1;
        string ans = "";
        int max_left=0, max_right=0;
        long len_max = 0;
        for(int i=0; i<=plus; i++){
            for(int j=0; j<len; j++){
                int left = j;
                int right = i+j;
                while((left>=0)&(right<len)){
                    // cout << s[left] << s[right] << endl;
                    if(s[left]==s[right]){
                        if(right-left+1>=len_max){
                            max_right=right;
                            max_left=left;
                            len_max = max_right - max_left + 1;
                        }
                        // cout << max_right << max_left << endl;
                        left--; right++;
                    }
                    else{
                        break;
                    }
                }
            }
        }
        ans = "";
        for(int pos=max_left;pos<max_right+1;pos++){
            ans.push_back(s[pos]);
        }
        return ans;
    }
    
    // 实现二分查找
    int higherBound(int nums[], int l, int r, int target)
    {
        // 在nums里找第一个大于等于target的下标
        int mid = -1;
        while(l < r){
            mid = (l + r) >> 1;
            if(nums[mid] <= target){
                l = mid + 1;
            }
            else{r = mid;}
        }
        return nums[l] > target ? l : -1;
    }
    int lowerBound(int nums[], int l, int r, int target)
    {
        // 在nums里找第一个大于等于target的下标
        int mid = -1;
        while(l < r){
            mid = (l + r) >> 1;
            if(nums[mid] < target){
                l = mid + 1;
            }
            else{r = mid;}
        }
        return nums[l] >= target ? l : -1;
    }
    
    //209. 长度最小的子数组
    int minSubArrayLen(int target, vector<int>& nums){
        vector<int> prefix;
        prefix.push_back(0);
        for(int num: nums){
            prefix.push_back(prefix[prefix.size() - 1] + num);
        }
        // prefix为前缀和
        int ans = nums.size();
        for(int i=1; i<=nums.size(); i++){
            int t = prefix[i - 1] + target;
            //            vector<int>::iterator idx = std::upper_bound(prefix.begin(), prefix.end(), t);
            int idx = std::upper_bound(prefix.begin(), prefix.end(), t) - prefix.begin();
            if(idx != prefix.size())
                ans = min(ans, idx - i + 1);
        }
        if(ans == nums.size())
            return 0;
        return ans;
    }
    
    // 46. 全排列
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> cur, visited(nums.size(), 0);
        permuteDFS(nums, ans, cur, visited);
        return ans;
    }
    void permuteDFS(vector<int>& nums, vector<vector<int>>& ans, vector<int>& cur, vector<int>& visited){
        int cnt = 0;
        for(int i: visited){
            cnt += i;
        }
        if(cnt == visited.size()){
            ans.push_back(cur);
            return;
        }
        for(int i=0; i<visited.size(); i++){
            if(visited[i] == 0){
                cur.push_back(nums[i]);
                visited[i] = 1;
                permuteDFS(nums, ans, cur, visited);
                visited[i] = 0;
                cur.pop_back();
            }
        }
    }
    // 31. 下一个排列
    void nextPermutation(vector<int> &nums){
        int n = nums.size();
        int i = n - 2;
        while(i >= 0 && nums[i] >= nums[i+1])
            i --;
        if(i >= 0){
            int j = n - 1;
            while(j >= i + 1 && nums[j] < nums[i])
                j--;
            swap(nums[i], nums[j]);
        }
        int left = i + 1, right = n - 1;
        while(left <= right){
            swap(nums[left], nums[right]);
            left++;
            right--;
        }
        //        sort(nums.begin() + i + 1,nums.end());
    }
    // 394. 字符串解码
    string decodeString(string s){
        stack<pair<string,int>> stk;
        string ans = "";
        string tmp = "";
        int multi = 0;
        for(int i=0; i<s.size(); i++){
            if((s[i] >= 'a' && s[i] <= 'z') ||(s[i] >= 'A' && s[i] <= 'Z')){
                ans.push_back(s[i]);
            }
            else if(s[i] >= '0' && s[i] <= '9'){
                multi = 10 * multi + s[i] - '0';
                cout << multi << endl;
            }
            else if(s[i] == '['){
                stk.push({ans, multi});
                ans = "";
                multi = 0;
            }
            else{
                //                pair<string, int> tmp = ;
                
                //                string cur_ans = ;
                //                ans += string(tmp.second, ans);
                for(int x=0; x<stk.top().second; x++){
                    stk.top().first += ans;
                }
                ans = stk.top().first;
                stk.pop();
            }
        }
        return ans;
    }
    
    // 1760. 袋子里最少数目的球
    int minimumSize(vector<int> & nums, int maxOperations)
    {
        int left = 1, right = *std::max_element(nums.begin(), nums.end());
        int ans = -1;
        // (nums[i]-1)/y，y是我们要二分寻找的数值
        while(left <= right){
            int mid = (right + left) >> 1;
            long long ops = 0;
            for(auto i: nums){ops += (i-1)/mid;}
            if(ops <= maxOperations){
                ans = mid;
                right = mid - 1;
            }
            else{left = mid + 1;}
        }
        return ans;
    }
    
    
    
    
    // 归并排序
    void guibingMerge(vector<int>& nums, int left, int right, int mid){
        int p1 = left, p2 = mid + 1;
        vector<int> tmp;
        while(p1 <= mid && p2 <= right){
            if(nums[p1] <= nums[p2]){
                tmp.push_back(nums[p1]);
                p1++;
            }
            else{tmp.push_back(nums[p2]);p2++;}
        }
        while(p1 <= mid){tmp.push_back(nums[p1]);p1++;}
        while(p2 <= right){tmp.push_back(nums[p2]);p2++;}
        for(int i = 0; i<tmp.size(); i++){
            nums[left + i] = tmp[i];
        }
    }
    void guibingMergeSort(vector<int>& nums, int left, int right){
        if(left == right){
            return ;
        }
        int mid = (right + left) / 2;
        guibingMergeSort(nums, left, mid);
        guibingMergeSort(nums, mid + 1, right);
        guibingMerge(nums, left, right, mid);
    }
    
    // 1750. 删除字符串两端相同字符后的最短长度

    int minimumLength(string s) {
        int n = s.size();
        int left = 0, right = n - 1;
        
        while(left < right && s[left] == s[right]){
            char cur = s[left];
            while(left <= right && s[left] == cur){
                left++;
            }
            while(left <= right && s[right] == cur){
                right--;
            }
        }
        return right - left + 1;
    }
    
//    单调栈 https://labuladong.github.io/algo/di-yi-zhan-da78c/shou-ba-sh-daeca/dan-diao-z-1bebe/
//    输入数组 nums = [2,1,2,4,3] 返回数组是每个位置上下一个最近的更大数，如果没有则为-1，结果数组为 [4,2,4,-1,-1]
//    数组倒着入栈。栈维护单调，栈顶要小于栈底，否则弹出目前的数字（因为新入栈的数对于未来入栈的数肯定是更接近的大数）。入栈时就判断结果是什么。
    int* nextLargerElement(int input[], int n)
    {
        stack<int> tmp;
        static int* res = new int[n];  // https://blog.csdn.net/qq_33185750/article/details/106978132
        for(int i = n - 1; i >= 0; i--){
            while(!tmp.empty() && tmp.top() <= input[i]){
                tmp.pop();
            }
            res[i] = tmp.empty() ? -1 : tmp.top();
            tmp.push(input[i]);
        }
        return res;
    }
    
    // "739. 每日温度"
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        // 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替
        // 输入: temperatures = [73,74,75,71,69,72,76,73]
        // 输出: [1,1,4,2,1,1,0,0]
        // 解法：
        //        逆序地遍历input数组。
        //        栈存储单调递减的序列，即每次比较栈顶，while（如果大于栈顶，则弹出栈顶），再把目前的数加到栈顶。
        //        如何确定加多少天？
        //            栈为空时，说明当前没有比目前数字更高的温度，应该是0.
        //            栈不为空，则目前栈顶是最近的更高温度的天，应该查看这是哪一天，并且返回gap，存储。
        //        因此，栈需要是 stack<int,int>
        int n = temperatures.size();
        vector<int> res(n);
        stack<pair<int, int>> tmp;
        for(int i=n-1; i>=0; i--){
            while(!tmp.empty() && temperatures[i]>=tmp.top().first){
                tmp.pop();
            }
            res[i] = tmp.empty() ? 0 : tmp.top().second - i;
            tmp.push(pair(temperatures[i], i));
        }
        return res;
    }
    
};


#endif /* Solution_hpp */
