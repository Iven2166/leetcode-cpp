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


class my_comp_listnode{
public:
    bool operator() (const ListNode* p1, const ListNode* p2){
        return p1->val > p2->val;
    }
};


struct ListNode{
    int val;
    ListNode* next;
    ListNode(): val(0), next(nullptr) {}
    ListNode(int x): val(x), next(nullptr) {}
    ListNode(int x, ListNode* next): val(x), next(next) {}
};

class Node{
public:
    int val;
    Node* left;
    Node* right;
    Node* parent;
};


struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(): val(0), left(nullptr), right(nullptr){}
    TreeNode(int x): val(x), left(nullptr), right(nullptr){}
    TreeNode(int x, TreeNode* left, TreeNode* right): val(x), left(left), right(right){}
};

class myPriorityQueueCompareOnListnodes
{
    bool reverse;
public:
    myPriorityQueueCompareOnListnodes(const bool &myparam=false){
        reverse = myparam;
    }
    bool operator() (const ListNode* p1, const ListNode* p2){
        return (p1->val > p2->val);
    }
};


class myPriorityQueueCompareOnInts
{
    bool reverse;
public:
    myPriorityQueueCompareOnInts(const bool& revparam=false)
    {
        reverse=revparam;
    }
    bool operator() (const int& lhs, const int& rhs){
        if(reverse){return (lhs>rhs);}
        else{return (lhs<rhs);}
    }
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
    /*
     给一个一堆括号的字符串，然后返回最长的合法的括号的长度。
     */
    int longestValidParentheses_1(string s) {
        int n = s.size();
        int res = 0;
        vector<int> dp(n, 0);
        for(int i=1; i<n; i++){
            if(s[i] == ')'){
                if(s[i-1] == '('){
                    dp[i] = (i-2>=0) ? dp[i-2] + 2 : 2;
                }
                else{
                    if(i - dp[i-1] - 1 >= 0 && s[i - dp[i-1] - 1] == '('){
                        dp[i] = dp[i-1] + 2;
                        if(i - dp[i-1] - 2 >= 0){
                            dp[i] += dp[i - dp[i-1] - 2];
                        }
                    }
                }
            }
            res = max(res, dp[i]);
        }
        return res;
    }
    
public:
    // 33. 搜索旋转排序数组
    /*
     整数数组 nums 按升序排列，数组中的值 互不相同 。

     在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
     给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
     你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
     --
     应该是二分法，不同的是，分段升序
     */
    int search33(vector<int>& nums, int target) {
        int n = nums.size();
        if(!n){
            return -1;
        }
        int left = 0, right = n - 1;
        while(left <= right){
            int mid = (right + left) / 2;
            if(target == nums[mid]){
                return mid;
            }
            if(nums[mid] >= nums[left]){
                if(target >= nums[left] && target < nums[mid]){
                    right = mid - 1;
                }
                else{
                    left = mid + 1;
                }
            }
            else {
                if(target > nums[mid] && target <= nums[right]){
                    left = mid + 1;
                }
                else{
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
    
    
    // 81. 搜索旋转排序数组 II
    
    bool search(vector<int>& nums, int target) {
        int n = nums.size();
        if(!n){
            return -1;
        }
        int left = 0, right = n - 1;
        while(left <= right){
            int mid = (right + left) / 2;
            if(target == nums[mid]){
                return true;
            }
            if(nums[left] == nums[mid] && nums[right]==nums[mid]){
                left++;
                right--;
            }
            else if(nums[mid] >= nums[left]){
                if(target >= nums[left] && target < nums[mid]){
                    right = mid - 1;
                }
                else{
                    left = mid + 1;
                }
            }
            else { // 注意这里不是 right 而是 n-1
                if(target > nums[mid] && target <= nums[n-1]){
                    left = mid + 1;
                }
                else{
                    right = mid - 1;
                }
            }
        }
        return false;
    }

    // 153. 寻找旋转排序数组中的最小值
    int findMin(vector<int>& nums) {
        int n = nums.size();
        if(n==1){
            return nums[0];
        }
        int left = 0, right = n-1;
        while(left < right){
            int mid = (right + left) / 2;
            if(nums[mid] < nums[right]){
                right = mid;
            }
            else{
                left = mid + 1;
            }
        }
        return nums[left];
    }
    
    // 978. 最长湍流子数组
    int maxTurbulenceSize(vector<int>& arr) {
        int n = arr.size();
        vector<vector<int>> dp(n, vector<int>(2,1));
        dp[0][0] = 1; dp[0][1] = 1;
        for(int i = 1; i < n; i++){
            if(arr[i] > arr[i-1]){
                dp[i][0] = dp[i-1][1] + 1;
            }
            else if(arr[i] < arr[i-1]) {
                dp[i][1] = dp[i-1][0] + 1;
            }
        }
        int res = 1;
        for(int i=0; i<n; i++){
            res = max(res, dp[i][0]);
            res = max(res, dp[i][1]);
        }
        return res;
    }
    
    int longestValidParentheses_2(string s) {
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
        int dp_last = INT_MIN, dp = 0;
        int res = INT_MIN;
        for(auto v: nums){
            dp = max(dp + v, v);
            res = max(res, dp);
        }
        return res;
    }
    
    /* 152. 乘积最大子数组
     给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。     测试用例的答案是一个 32-位 整数。
     子数组 是数组的连续子序列。

     输入: nums = [2,3,-2,4]
     输出: 6
     解释: 子数组 [2,3] 有最大乘积 6。
     链接：https://leetcode.cn/problems/maximum-product-subarray
     
     维护一个最大最小的结果？
    */
    int maxProduct(vector<int>& nums) {
        int res = INT_MIN;
        int n = nums.size();
        if(n==1)
            return nums[0];
        int dp1 = nums[0];
        int dp2 = nums[0];
        res = max(dp1, dp2);
        for(int i=1; i<n; i++){
            int tmp = dp1;
            dp1 = min(dp1 * nums[i], dp2 * nums[i]);
            dp1 = min(dp1, nums[i]);
            dp2 = max(tmp * nums[i], dp2 * nums[i]);
            dp2 = max(dp2, nums[i]);
            res = max(max(res, dp2), dp1);
        }
        return res;
    }
    
    // 697. 数组的度
    
    int findShortestSubArray(vector<int>& nums) {
        unordered_map<int, vector<int>> dict;
        for(int i=0; i<nums.size(); i++){
            if(!dict.count(nums[i])){
                // find none
                dict[nums[i]] = {1, i, i};
            }
            else{
                dict[nums[i]][0]++;
                dict[nums[i]][2] = i;
            }
        }
        int res = 0;
        int num_max = 0;
        for(auto it=dict.begin(); it!=dict.end(); it++){
            // 可以计算目前数字的最长范围 it->second[2] - it->second[1] + 1， 和目前的进行比较
            
            if(it->second[0] == num_max){
                // 数字是当前具备最大度
                res = min(res, it->second[2] - it->second[1] + 1);
            }
            else if(it->second[0] > num_max){
                num_max = it->second[0];
                res = it->second[2] - it->second[1] + 1; // 必须覆写
            }
        }
        return res;
    }
    
    // 55. 跳跃游戏
    bool canJump_method1(vector<int>& nums) {
        // 复杂写法：时间复杂度 O(N^2)
        int n = nums.size();
        vector<int> dp(n, 0);
        dp[0] = 1;
        for(int i=0; i<n; i++){
            if(dp[i] > 0){
                for(int j=i; j<=min(n-1, i+nums[i]); j++){
                    dp[j] = 1;
                }
            }
            if(dp[n-1] > 0){
                return true;
            }
        }
        return dp[n-1] > 0;
    }
    
    bool canJump_method2(vector<int>& nums) {
        // 简化写法: 时间复杂度 O(N^2)
        // 实际只需要记录最长可抵达下标即可
        int n = nums.size();
        int largest = 0; // 目前是在下标0
        for(int i=0; i<n; i++){
            if(i <= largest){
                largest = (i + nums[i] > largest) ? i + nums[i]: largest;
                if(largest >= n - 1){
                    return true;
                }
            }
        }
        return false;
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
    
    // 148. 排序链表
    // 堆排序 + 重建
    ListNode* sortList_method1(ListNode* head){
        priority_queue<int, vector<int>, greater<int>> q;
        ListNode* tmp = head;
        while(tmp){
            q.push(tmp->val);
            tmp = tmp->next;
        }
        ListNode* dummy = new ListNode(0);
        tmp = dummy;
        while(!q.empty()){
            int v = q.top();
            q.pop();
            tmp->next = new ListNode(v);
            tmp = tmp->next;
        }
        return dummy->next;
    }
    // 归并排序
    ListNode* sortList_method2(ListNode* head){
        // 归并排序：1、双指针，确定中心位置 2、针对前后分别调用 sortList 3、合并两个结果
        // base case 开始
        if(head == nullptr || head->next == nullptr)
            return head;
        // base case 结束
        
        // 快慢指针
        ListNode* p1 = head;
        ListNode* p2 = head->next;
        
        while(p1 && p2 && p2->next){
            p1 = p1->next;
            p2 = p2->next->next;
        }
        ListNode* half2 = p1->next;
        p1->next = nullptr;
        ListNode* half1 = head;
        
        // 分别排序
        half1 = sortList_method2(half1);
        half2 = sortList_method2(half2);
        
        // 合并
        ListNode* dummy = new ListNode(0);
        ListNode* cur = dummy;
        int nxt = 0;
        while(half1 && half2){
            if(half1->val < half2->val){
                nxt = half1->val;
                half1 = half1->next;
            }
            else{
                nxt = half2->val;
                half2 = half2->next;
            }
            cur->next = new ListNode(nxt);
            cur = cur->next;
        }
        if(half1 || half2)
            cur->next = half1 ? half1 : half2;
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
    /*
     给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
     输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
     输出：6
     解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。da

     来源：力扣（LeetCode）
     链接：https://leetcode.cn/problems/trapping-rain-water
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
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
    
    // 我觉得这一版写的更清晰，先定义起点，然后做二分查找
    vector<vector<int>> threeSumBetter(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        for(int start=0; start<nums.size()-2; start++){
            if((nums[start] > 0) || (nums[start]+nums[start+1]+nums[start+2] > 0)){ // 剪枝: 60%->85%
                return res;
            }
            if(start>=1 && nums[start]==nums[start-1]){
                continue;
            }
            int target = - nums[start];
            int left = start + 1;
            int right = nums.size() - 1;
            while(left < right){
                if(nums[left] + nums[right] == target){
                    res.push_back({nums[start],nums[left],nums[right]});
                    // 可能还会有其他的组合
                    while(left<right && nums[left]==nums[left+1]){
                        left++; // 如果遇到相同的数字，就继续往前
                    }
                    while(left<right && nums[right]==nums[right-1]){
                        right--; // 如果遇到相同的数字，就继续往前
                    }
                    left++;
                    right--;
                }
                else if(nums[left] + nums[right] > target){
                    right--;
                }
                else{
                    left++;
                }
            }
        }
        return res;
    }
    
    //259. 较小的三数之和
    // 双指针
    int threeSumSmaller_method1(vector<int>& nums, int target) {
        if(nums.size()==0){return 0;}
        if(nums.size()<3){return 0;}
        sort(nums.begin(), nums.end());
        int ans = 0;
        for(int i=0; i<nums.size()-2; i++){
            int left = i+1;
            int right=nums.size()-1;
            // cout << i<< ','<< left << ',' << right<<endl;
            while(left < right){
                if(nums[i] + nums[left] + nums[right] < target){
                    ans+=(right-left);
                    left++;
                }
                else{
                    right--;
                }
            }
        }
        return ans;
    }
public:
    // 方法2: 二分查找
    int threeSumSmaller_method2(vector<int>& nums, int target) {
        if(nums.size()<3){return 0;}
        sort(nums.begin(), nums.end());
        int res = 0;
        for(int i=0; i<nums.size()-2; i++){
            res += twoSumSmaller(nums, i+1, target-nums[i]);
        }
        return res;
    }
private:
    int twoSumSmaller(vector<int>& nums, int startIdx, int target){
        int res = 0;
        for(int i=startIdx; i<nums.size()-1; i++){
            int pos = binarySearch(nums, i, target-nums[i]);
            res += (pos - i);
        }
        return res;
    }
    int binarySearch(vector<int>& nums, int startIdx, int target){
        // 寻找 nums[startIdx:] 之间小于target的最大数
        int left = startIdx;
        int right = nums.size() - 1;
        while(left < right){
            int mid = (right + left + 1) / 2; // 注意需要加1，因为是左右闭区间
            if(nums[mid] >= target){
                right = mid - 1; // mid 不可能，所以更换为 mid-1
            }
            else if(nums[mid] < target){
                left = mid; // 别超过小于target的最大数
            }
        }
        return left;
    }
    
    //17. 电话号码的字母组合
private:
    unordered_map<char, string> letterCombinations_map;
    string letterCombinations_tmp;
public:
    vector<string> letterCombinations(string digits){
        /*
         是一颗多叉树，到无子节点的节点时，将经过路径的所有节点，结合为字符串
         */
        if(digits=="")
            return {};
        letterCombinations_map['2'] = "abc";
        letterCombinations_map['3'] = "def";
        letterCombinations_map['4'] = "ghi";
        letterCombinations_map['5'] = "jkl";
        letterCombinations_map['6'] = "mno";
        letterCombinations_map['7'] = "pqrs";
        letterCombinations_map['8'] = "tuv";
        letterCombinations_map['9'] = "wxyz";
        vector<string> res;
        letterCombinationsHelper(digits, res, 0);
        return res;
    }
    
    void letterCombinationsHelper(string digits,
                                  vector<string>& res,
                                  int pos){
        if(pos == digits.size()){
            res.emplace_back(letterCombinations_tmp);
            return;
        }
        char cur = digits[pos];
        for(int i=0; i<letterCombinations_map[cur].size(); i++){
            letterCombinations_tmp.push_back(letterCombinations_map[cur][i]);
            letterCombinationsHelper(digits, res, pos+1);
            letterCombinations_tmp.pop_back();
        }
    }
    
public:
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
    int lengthOfLongestSubstring1(string s) {
//        if(s=="")
//            return 0;
//        // 从左到右遍历。用一个hashmap保存当前字符出现的最新位置 pos（应该是最右边）。查找当下位置i对应字符的pos，如果大于start，说明start应该更新为 pos+1；反之不理。pos所在字符的map-value更新为i
//        unordered_map<char, int> hashmap;
//        int start = 0;
//        hashmap[s[0]] = 0;
//        int len_max = 1;
//        for(int i=1; i<s.size(); i++){
//            auto iter = hashmap.find(s[i]);
//            if((iter!=hashmap.end())){
//                if( (iter->second >= start)){
//                    start = iter->second + 1;
//                }
//            }
//            hashmap[s[i]] = i;
//            if(i-start+1>len_max)
//                len_max = i - start + 1;
//        }
//        return len_max;
        
        if(s=="") return 0;
        int left = 0, res = INT_MIN;
        unordered_map<char, int> map;
        for(int right = 0; right < s.size(); right++){
            if(map.count(s[right]) && map[s[right]] >= left){
                left = map[s[right]] + 1;
            }
            map[s[right]] = right;
            res = max(res, right - left + 1);
        }
        return res;
    }
    
    //3. 无重复字符的最长子串
    /*
     s = "abcabcbb", 有一个map记录移动窗口里，每个字母的出现次数。
     用while进行，如果right处有字母出现2次，则字典里减去窗口的left所占字母，并且left向右移动，直到right所占字母的次数=1为止
     全程的res=max(res, right-left)
     */
    int lengthOfLongestSubstring2(string s){
        unordered_map<char, int> str_cnt;
        int left = 0, right = 0;
        int res = 0;
        int len = s.size();
        while(right < len){
            // c 是即将进入窗口的字符
            char c = s[right];
            // 窗口右开侧准备右翼
            right++;
            // 进行窗口内数据的更新
            if(str_cnt.count(c)){
                str_cnt[c]++;
            }
            else{str_cnt[c] = 1;}
            // 开始判断是否需要向右收缩窗口
            while(str_cnt[c] > 1){
                str_cnt[s[left]]--;
                left++;
            }
            // 合法的窗口
            res = max(res, right - left);
        }
        return res;
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
        /*
         我们希望找到一种方法，能够找到一个大于当前序列的新序列，且变大的幅度尽可能小。具体地：我们需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大，从而得到下一个排列。同时我们要让这个「较小数」尽量靠右，而「较大数」尽可能小。当交换完成后，「较大数」右边的数需要按照升序重新排列。这样可以在保证新排列大于原来排列的情况下，使变大的幅度尽可能小。
         链接：https://leetcode.cn/problems/next-permutation/solution/xia-yi-ge-pai-lie-by-leetcode-solution/
         以 [4,5,2,6,3,1] 为例：
         1、从右到左，找到第一个顺序对 [2,6]，定义 2 位置为 i，则 [i+1,n)为降序排列
         2、从右到左，找到第一个大于 nums[i] = 2的数，即 3，位置j
         3、交换 i j 位置，成为  [4,5,3,6,2,1]
         4、此刻 [i+1, n) 为降序，用双指针改为 升序，完成 [4,5,3,1,2,6]
         */
        int n = nums.size();
        int i = n - 2;
        while(i >= 0 && nums[i] >= nums[i+1])
            i --;
        // 此时 i 右侧为降序
        
        // 从右侧寻找一个刚好大于 nums[i] 的位置
        if(i >= 0){
            int j = n - 1;
            while(j >= i + 1 && nums[j] <= nums[i])
                j--;
            swap(nums[i], nums[j]);
        }
        // sort(nums.begin() + i + 1,nums.end());
        int left = i + 1, right = n - 1;
        while(left <= right){
            swap(nums[left], nums[right]);
            left++;
            right--;
        }
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
    
    //    "503. 下一个更大元素 II"
    //    给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），返回 nums 中每个元素的 下一个更大元素 。
    //
    //    数字 x 的 下一个更大的元素 是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1 。
    //
    //    来源：力扣（LeetCode）
    //    链接：https://leetcode.cn/problems/next-greater-element-ii
    //    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    vector<int> nextGreaterElement2(vector<int> & nums){
        vector<int> input = nums;
        input.insert(input.end(), nums.begin(), nums.end());
        vector<int> output(nums.size());
        int n = nums.size();
        stack<int> tmp;
        for(int i=input.size()-1; i>=0; i--){
            while(!tmp.empty() && input[i]>=tmp.top())
                tmp.pop();
            output[i%n] = tmp.empty()? -1: tmp.top();
            tmp.push(input[i]);
        }
        return output;
    }
    
    //    "698. 划分为k个相等的子集"
    //    给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。
    //    参考：https://labuladong.github.io/algo/di-san-zha-24031/bao-li-sou-96f79/jing-dian--93320/
    //    解法：
    //        对于每一个数字来说，应该选择哪个桶。所以回溯backtrack写在数字的角度。
    
    bool canPartitionKSubsets698(vector<int> & input, int k){
        
        int sum = 0;
        for(int v: input){sum += v;}
        if(sum % k > 0){return false;}
        int target = sum / k;
        vector<int> bucket(k); //
        
        return canPartitionKSubsets_backtrack(input, 0, target, bucket);
    }
    
    bool canPartitionKSubsets_backtrack(vector<int>& input, int index, int target, vector<int>& bucket){
        if(index == input.size()){
            //            到达最后一位数字的选择，此时如果已经选完，那么全部bucket应该是刚好填满
            for(int i=0; i<bucket.size(); i++){
                if(bucket[i]!=target){
                    return false;
                }
            }
            return true;
        }
        // 穷举 input[index] 目前能够装入的桶
        for(int i = 0; i < bucket.size(); i++){
            // 将 input[index] 装入目前的桶里
            bucket[i] += input[index];
            if(canPartitionKSubsets_backtrack(input, index + 1, target, bucket)){
                return true;
            }
            bucket[i] -= input[index];
        }
        // input[index]无法装入任何一个桶
        return false;
    }
    
    // 从桶的视角，来进行数字的放入与不放入的选择。时间复杂度：O(k* 2^N) 对于k个桶来说，遍历n个数字，做2次选择（放入与不放入）
    bool canPartitionKSubsets698_fromBuckets(vector<int> & input, int k){
        int sum = 0;
        for(int v: input){sum += v;}
        if(sum % k > 0){return false;}
        int target = sum / k;
        int used = 0;
        unordered_map<int, bool> memo; // 用以记录中间状态，减少冗余的回溯
        // k 号桶初始什么都没装，从 nums[0] 开始做选择
        return canPartitionKSubsets698_fromBuckets_backtrack(input, 0, 0, used, target, k, memo);
    }
    
    bool canPartitionKSubsets698_fromBuckets_backtrack(vector<int>& input,
                                                       int bucket_value,
                                                       int start,
                                                       int used,
                                                       int target,
                                                       int k,
                                                       unordered_map<int, bool> memo){
        if(k == 0){return true; } // 所有桶都被装满，因为sum%k==0
        if(bucket_value == target){
            // 装满了这个桶,到下一个桶
            bool res = canPartitionKSubsets698_fromBuckets_backtrack(input, 0, 0, used, target, k-1, memo);
            memo[used] = res;
            return res;
        }
        if(memo.count(used)){return memo[used];}
        for(int i = start; i < input.size(); i++){
            if(((used>>i) & 1) == 1){continue;}//i位置已经装入别的桶里
            if(input[i] + bucket_value > target){continue;}
            
            used |= (1<<i);
            bucket_value += input[i];
            if(canPartitionKSubsets698_fromBuckets_backtrack(input, bucket_value, i + 1, used, target, k, memo)){return true;}
            used ^= (1<<i);
            bucket_value -= input[i];
        }
        return false;
    }
    
    //    "86. 分隔链表"
    //    给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。
    //    你应当 保留 两个分区中每个节点的初始相对位置。
    //    来源：力扣（LeetCode）
    //    链接：https://leetcode.cn/problems/partition-list
    //    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    ListNode* partition86(ListNode* head, int x){
        // cur 指向 head，新建两个 listnode，分别用p1和p2来指向. 如果x值出现多次，如何保存？
        ListNode* curr = head;
        ListNode* dummy1 = new ListNode(0);
        ListNode* dummy2 = new ListNode(0);
        ListNode* dummy3 = new ListNode(0);
        ListNode* p1 = dummy1;
        ListNode* p2 = dummy2;
        ListNode* p3 = dummy3;
        
        while(curr != nullptr){
            if(curr->val >= x){
                p3->next = new ListNode(curr->val);
                p3 = p3->next;
            }
            else if (curr->val < x){
                p1->next = new ListNode(curr->val);
                p1 = p1->next;
            }
            else{
                p2->next = new ListNode(curr->val);
                p2 = p2->next;
            }
            curr = curr->next;
        }
        // 到了末尾，开始拼接
        p2->next = dummy3->next;
        p1->next = dummy2->next;
        return dummy1->next;
    }
    
    // 23. 合并K个升序链表
    //    给你一个链表数组，每个链表都已经按升序排列。
    //
    //    请你将所有链表合并到一个升序链表中，返回合并后的链表。
    //    输入：lists = [[1,4,5],[1,3,4],[2,6]]
    //    输出：[1,1,2,3,4,4,5,6]
    //    解释：链表数组如下：
    //    [
    //      1->4->5,
    //      1->3->4,
    //      2->6
    //    ]
    //    将它们合并到一个有序链表中得到。
    //    1->1->2->3->4->4->5->6
    //
    //    来源：力扣（LeetCode）
    //    链接：https://leetcode.cn/problems/merge-k-sorted-lists
    //    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    
    
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if(lists.size() == 0){return nullptr;}
        priority_queue<ListNode*, vector<ListNode*>, my_comp_listnode> q;
        for(auto value: lists){
            if(value!=nullptr)
                q.push(value);
        }
        ListNode* res = new ListNode(0);
        ListNode* dummy = res;
        while(!q.empty()){
            ListNode* tmp = q.top();
            q.pop();
            dummy->next = new ListNode(tmp->val);
            dummy = dummy->next;
            tmp = tmp->next;
            if(tmp!=nullptr){q.push(tmp);}
        }
        return res->next;
    }
    
    // 方法2:分治算法
    ListNode* mergeKLists_fenzhi(vector<ListNode*>& lists) {
        /*
        分治算法：这个vector有n长度，采用二分法逐步找到最底层，再合并。
        最底层是实现两个 排序listnode 合并的算法
        */
        ListNode* res = mergeKListsBetween(lists, 0, lists.size() - 1);
        return res;
    }
    
    ListNode* mergeKListsBetween(vector<ListNode*>& lists, int l, int r){
        // base case
        if(l == r){
            return lists[l];
        }
        if(l > r){
            return nullptr;
        }
        // base case end
        int mid = (r + l) / 2;
        return mergeTwoListNode(mergeKListsBetween(lists, l, mid),
                                mergeKListsBetween(lists, mid + 1, r));
    }
    
    ListNode* mergeTwoListNode(ListNode* l1, ListNode* l2){
        if(l1 == nullptr || l2 == nullptr){
            return (l1 != nullptr) ? l1 : l2;
        }
        // both two listnodes are not nullptr
        ListNode *p1 = l1, *p2 = l2;
        ListNode* res = new ListNode(0);
        ListNode* dummy = res;
        while(p1 && p2){
            if(p1->val < p2->val){
                dummy->next = new ListNode(p1->val);
                p1 = p1->next;
            }
            else{
                dummy->next = new ListNode(p2->val);
                p2 = p2->next;
            }
            dummy = dummy->next;
        }
        dummy->next = (p1!=nullptr) ? p1 : p2;
        return res->next;
    }
    
    // 19. 删除链表的倒数第 N 个结点
    // 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        
        ListNode* p1 = dummy;
        for(int i=0; i<n; i++){p1 = p1->next;}
        ListNode* p2 = dummy;
        while(p1->next!=nullptr){p2=p2->next; p1=p1->next;}
        p2->next = p2->next->next;
        return dummy->next;
    }
    
    ListNode* removeNthFromEnd_method2(ListNode* head, int n){
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* p = dummy;
        
        stack<ListNode*> stk;
        while(p!=nullptr){
            stk.push(p);
            p = p->next;
        }
        while(n>0){
            stk.pop();
            n--;
        }
        ListNode* end_p = stk.top();
        end_p->next = end_p->next->next;
        return dummy->next;
    }
    
    // 876. 链表的中间结点
    ListNode* middleNode(ListNode* head){
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* p1 = dummy;
        ListNode* p2 = dummy;
        while(p1!=nullptr){
            if(p1->next!=nullptr){p1 = p1->next->next;p2 = p2->next;}
            else{
                p2 = p2->next;
                break;
            }
        }
        return p2;
    }
    
    // 141. 环形链表
    bool hasCycle_method1(ListNode *head) {
        ListNode* p1 = head;
        ListNode* p2 = head;
        while(p1!=nullptr && p1->next!=nullptr){
            p1 = p1->next->next;
            p2 = p2->next;
            if(p1==p2){
                return true;
            }
        }
        // 一旦为空，说明为非环
        return false;
    }
    
    bool hasCycle_method2(ListNode *head) {
        unordered_set<ListNode*> seen;
        while(head){
            if(seen.count(head)){
                return true;
            }
            seen.insert(head);
            head = head->next;
        }
        return false;
    }
    
    //    "142. 环形链表 II"
    ListNode *detectCycle(ListNode *head) {
        ListNode* p1 = head;
        ListNode* p2 = head;
        while(p1!=nullptr && p1->next!=nullptr){
            p1 = p1->next->next;
            p2 = p2->next;
            if(p1 == p2)
            {
                ListNode* p3 = head;
                // 此刻p2走了k步，p1走了2k步；交叉点距离环入口 m 步。此刻让 p2 再走，新的从头开始
                while(p3!=nullptr && p2!=nullptr){
                    if(p2 == p3){return p2;}
                    p3 = p3->next;
                    p2 = p2->next;
                }
            }
        }
        return nullptr;
    }
    
    // "160. 相交链表"
    ListNode *getIntersectionNode_method1(ListNode *headA, ListNode *headB) {
        unordered_set<ListNode*> seen;
        ListNode* p1 = headA;
        ListNode* p2 = headB;
        while(p1!=NULL || p2!=NULL){
            if(p1){
                if(seen.count(p1)){return p1;}
                seen.insert(p1);
                p1 = p1->next;
            }
            if(p2){
                if(seen.count(p2)){return p2;}
                seen.insert(p2);
                p2 = p2->next;
            }
        }
        return nullptr;
    }
    
    ListNode *getIntersectionNode_method2(ListNode *headA, ListNode *headB) {
            ListNode* p1 = headA;
            ListNode* p2 = headB;
            int c1 = 0;
            int c2 = 0;
            while(p1!=nullptr || p2!=nullptr){
                if(p1 == p2){return p1;}
                p1 = p1->next;
                p2 = p2->next;
                if(p1==nullptr && c1==0){p1 = headB; c1++;}
                if(p2==nullptr && c2==0){p2 = headA; c2++;}
            }
            return nullptr;
    }
    //    26. 删除有序数组中的重复项
    int removeDuplicates(vector<int>& nums){
        int n = nums.size();
        if(n==0)return 0;
        int slow = 0, fast = 0;
        
        while(fast < n){
            if(nums[slow] != nums[fast]){
                slow++;
                nums[slow] = nums[fast];
            }
            fast++;
        }
        return slow;
    }
    
    // 83. 删除排序链表中的重复元素
    ListNode* deleteDuplicates83(ListNode* head) {
        if(head == nullptr){return head;}
        ListNode* slow = head;
        ListNode* fast = head;
        while(fast != nullptr){
            if(fast->val != slow->val){
                slow = slow->next;
                slow->val = fast->val;
            }
            fast = fast->next;
        }
        slow->next = nullptr;
        return head;
    }
    
    // 283. 移动零
    void moveZeroes(vector<int>& nums) {
        int n = nums.size();
        // remove all zeros
        int slow = 0, fast = 0;
        while(fast < n){
            if(nums[fast] != 0){
                nums[slow] = nums[fast];
                slow++;
            }
            fast++;
        }
        // add remain pos 0
        for(int i = slow; i < n; i++){nums[i] = 0;}
    }
    
    //    1625. 执行操作后字典序最小的字符串
    string findLexSmallestString(string s, int a, int b) {
        // 枚举
        int n = s.size();
        vector<int> vis(n);
        string res = s;
        s = s + s; // 类似回文的操作，便于查找
        
        for(int i = 0; vis[i] == 0; i = (i + b) % n){
            vis[i] = 1;
            for(int j = 0; j < 10; j++){
                int k_limit = b % 2 == 0 ? 0 : 9;
                for(int k = 0; k <= k_limit; k++){
                    string tmp = s.substr(i, n);
                    for(int p = 1; p < n; p += 2){
                        tmp[p] = '0' + (tmp[p] - '0' + j * a) % 10;
                    }
                    for(int p = 0; p < n; p += 2){
                        tmp[p] = '0' + (tmp[p] - '0' + k * a) % 10;
                    }
                    res = min(tmp, res);
                }
            }
        }
        
        return res;
    }
    
    // 76. 最小覆盖子串
    /*
     输入：s = "ADOBECODEBANC", t = "ABC"
     输出："BANC"
     解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。

     来源：力扣（LeetCode）
     链接：https://leetcode.cn/problems/minimum-window-substring
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    string minWindow(string s, string t){
        unordered_map<char, int> window, need;
        int left = 0, right = 0;
        int valid = 0;  // 用以记录当前窗口满足的覆盖字母去重个数
        int start = 0, len = INT_MAX;
        for(char i: t){need[i]++;}
        while(right < s.size()){
            // 即将进入窗口的字符
            char c = s[right];
            right++;
            // 开始更新窗口状态,只更新我们关心的字母
            if(need.count(c)){
                window[c]++;
                if(window[c]==need[c]) // 如果相等，仅会发生一次。加法无需判断，减法（left右移情况）会单独判断
                    valid++;
            }
            // 开始判断是否能够缩小窗口，右移left
            while(need.size()==valid){
                // 由于窗口已经满足条件，在这里更新最小覆盖区间的答案
                if(right - left < len){
                    len = right - left;
                    start = left;
                }
                // 即将离开窗口的字符
                char d = s[left];
                left++;
                // 离开后判断这个窗口是否会有关心字母的减少
                // 注意**：这里是跟上述更新窗口状态反过来的
                if(need.count(d)){
                    if(window[d]==need[d])
                        valid--;
                    window[d]--;
                }
            }
        }
        return len==INT_MAX ? "" : s.substr(start, len);
    }
    
    // 567. 字符串的排列
    bool checkInclusion(string t, string s) {
        unordered_map<char, int> window, need;
        int left = 0, right = 0;
        int valid = 0;
        for(auto i: t){need[i]++;}
        while(right < s.size()){
            char c = s[right];
            right++;
            // 判断窗口状态
            if(need.count(c)){
                window[c]++;
                if(window[c]==need[c]) valid++;
            }
            // 判断是否需要收缩窗口，保持定长的窗口 t.size()
            while(right - left >= t.size()){
                if(valid==need.size()) return true; // 这其实发生于 right - left == t.size()
                // 否则应该收缩窗口
                char d = s[left];
                left++;
                if(need.count(d)){
                    if(window[d]==need[d]) valid--;
                    window[d]--;
                }
            }
        }
        return false;
    }
    
    // 167. 两数之和 II - 输入有序数组
    vector<int> twoSum167(vector<int>& numbers, int target) {
        int left = 0, right = numbers.size() - 1;
        int res1 = -1, res2 = -1;
        while(left < right){
            int sum = numbers[left] + numbers[right];
            if(sum == target){
                return {left+1, right+1};
            }
            else if(sum > target){right--;}
            else{left++;}
        }
        return {-1,-1};
    }
    
    //5. 最长回文子串
    string longestPalindrome5(string s) {
        string res = "";
        for(int i=0; i<s.size(); i++){
            string res1 = longestPalindromeCore(s, i, i);
            string res2 = longestPalindromeCore(s, i, i+1);
            res = (res.size() < res1.size()) ? res1 : res;
            res = (res.size() < res2.size()) ? res2 : res;
        }
        return res;
    }
    
    // 先查找以中心向两端扩散的字符串
    string longestPalindromeCore(string s, int l, int r){
        while(l >= 0 && r <= s.size() - 1 && s[l]==s[r]){
            l--; r++;
        }
        // 因为在最后一次不符合while内条件时，l比正确位置多-1，r多+1
        return s.substr(l + 1, r - l - 1);
    }
    
public:
    vector<vector<string>> res;
    deque<string> track;  // 路径存放
    bool ifPalindrome(string s, int l, int r){
        while(l <= r){
            if(s[l]==s[r]){l++; r--;}
            else{return false;}
        }
        return true;
    }
    vector<vector<int>> partition131DPmap_init(string s){
        vector<vector<int>> partition131DPmap(s.size(), vector<int>(s.size(), 1));
        for(int i = s.size() - 1; i > -1; i--){
            for(int j = i + 1; j < s.size(); j++){
                if(j < s.size() && s[i] == s[j] && partition131DPmap[i+1][j-1] == 1){
                    partition131DPmap[i][j] = 1;
                }
                else{
                    partition131DPmap[i][j] = 0;
                }
            }
        }
        return partition131DPmap;
    }
    void partition131_backtrack(string s, int start){
        if(start == s.size()){
            // 到达叶子节点，将答案放到res里
            res.push_back(vector<string>(track.begin(), track.end()));
            return ;
        }
        // 还有之后的节点要走
        for(int i = start; i < s.size(); i++){
            if(!ifPalindrome(s, start, i)){
                continue;
            }
            // s[start:i] 是一个回文串（可以分割？）
            // 做选择
            track.push_back(s.substr(start, i - start + 1));
            // 进入下一层：继续切分 s[i+1:]
            partition131_backtrack(s, i+1);
            track.pop_back();
        }
    }
    
public:
    // 131. 分割回文串
    vector<vector<string>> partition131(string s) {
        partition131_backtrack(s, 0);
        return res;
    }

    // 658. 找到 K 个最接近的元素
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        // 二分查找: arr 分为 [0,left] < x，[right,n-1]>x
        // 然后再双指针查找两侧
        int right = lower_bound(arr.begin(), arr.end(), x) - arr.begin();
        int left = right - 1;
        while(k--){
            if(left < 0){
                right++; // 左指针超过边界
            }
            else if(right >= arr.size()){
                left--; //右指针超过边界
            }
            else if(x - arr[left] <= arr[right] - x){
                left--;
            }
            else{
                right++;
            }
        }
        vector<int> res = vector<int>(arr.begin() + left + 1, arr.begin() + right);
        return res;
    }
    
private:
    bool findClosestElements_CompareLarger(int a, int b, int x){
        return (abs(a - x) < abs(b - x) || (a < b && abs(a - x)==abs(b - x)));
    }
public:
    // 658. 找到 K 个最接近的元素 --->看下自己写的复杂程度！！！
    vector<int> findClosestElements_wrotemyself(vector<int>& arr, int k, int x) {
        /* 有点像寻找最长回文子串？
         1、找到“中心点”：双指针，left和right，如果相同说明应该是本身，如果不是则是两个选择
         2、向左右扩散，并且比较左右指针的大小（符合规则），或者边界，来决定如何移动指针
         3、arr的两个指针范围，应该是连续的子数组
        */
        int left = 0, right = arr.size() - 1;
        int begin_left = 0, begin_right = arr.size() - 1;
        int left_gap_min = INT_MAX, right_gap_min = INT_MAX;
        int left_gap = 0, right_gap = 0;
        while(left <= right){
            left_gap = abs(x - arr[left]);
            right_gap = abs(x - arr[right]);
            if(left_gap_min > left_gap){
                left_gap_min = left_gap;
                begin_left = left;
                left++;
            }
            if(right_gap_min > right_gap){
                right_gap_min = right_gap;
                begin_right = right;
                right--;
            }
        }
        vector<int> res(k, 0);
        if(k==1){
            if(begin_left==begin_right){return {arr[begin_left]};}
            else{
                if(findClosestElements_CompareLarger(arr[begin_left],arr[begin_right],x)){
                    return {arr[begin_left]};
                }
                return {arr[begin_right]};
            }
        }
        int res_left = 0, res_right = arr.size() - 1;
        int l = begin_left, r = begin_right;
        while(k > 0){
            if(l == r){
                res_left = l; res_right = r;
                l--; r++; k--;}
            else{
                if(findClosestElements_CompareLarger(arr[l], arr[r], x)){
                    res_left = l;
                    l--; k--;
                }
                else{
                    res_right = r;
                    r++; k--;
                }
            }
        }
        int pos = res_left;
        for(int i=0; i<res.size(); i++){
            res[i] = arr[pos];
            pos++;
        }
        return res;
//        return {arr[begin_left], arr[begin_right]};
    }
    
    //  82. 删除排序链表中的重复元素 II
    ListNode* deleteDuplicates82(ListNode* head) {
        ListNode* dummy = new ListNode(-101, head);
        ListNode* p = dummy;
        while(p->next && p->next->next){
            if(p->next->val == p->next->next->val){
                int x = p->next->val;
                while(p->next && p->next->val == x){
                    p->next = p->next->next;
                }
            }
            else{
                p = p->next;
            }
        }
        return dummy->next;
    }
    
    // 543. 二叉树的直径
    int diameterOfBinaryTreeRes = 0;
    int diameterOfBinaryTreeBase(TreeNode* root){
        if(root==nullptr) return 0;
        int left = diameterOfBinaryTreeBase(root->left);
        int right = diameterOfBinaryTreeBase(root->right);
        diameterOfBinaryTreeRes = (left + right > diameterOfBinaryTreeRes) ? left + right : diameterOfBinaryTreeRes;
        return 1 + max(left, right);
    }
    int diameterOfBinaryTree(TreeNode* root) {
        // 每个节点左右总共有多少个节点
        diameterOfBinaryTreeBase(root);
        return diameterOfBinaryTreeRes;
    }
    
    //236. 二叉树的最近公共祖先
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root==nullptr){return nullptr;}
        if(root == p || root == q){
            return root;
        }
        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);
        if(left != nullptr && right != nullptr){return root;} // 一个在左，一个在右
        if(left != nullptr){return left;}
        if(right != nullptr){return right;}
        return nullptr;
    }
    
    // 1650. 二叉树的最近公共祖先 III
    Node* lowestCommonAncestor(Node* p, Node * q) {
        unordered_set<int> set;
        while(p){
            set.insert(p->val);
            p = p->parent;
        }
        while(q){
            if(set.find(q->val) != set.end()){
                return q;
            }
            q = q->parent;
        }
        return nullptr;
    }
    
    // 322. 零钱兑换
    int coinChange(vector<int>& coins, int k){
        if(k==0){return 0;}
        // 比如 coins = {1,3,5}, k=11，则要么先到达 11-1=10, 11-3=8,11-5=6，这里消耗了一个硬币，再进行细拆
        int max = k + 1;
        vector<int> dp(k+1, max);
        dp[0] = 0;
        // for(auto i: coins){dp[i] = 1;} // 一枚硬币即可
        for(int i = 0; i < k+1; i++){
            for(auto c: coins){
                if(i >= c) dp[i] = min(dp[i], dp[i-c] + 1);
            }
        }
        return  (dp[k] > k)? -1: dp[k];
    }
    
    // 39. 组合总和
    /*
     candidates = [2,3,6,7], target = 7
     给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
     candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。
     对于给定的输入，保证和为 target 的不同组合数少于 150 个。
     示例 1：
     输入：candidates = [2,3,6,7], target = 7
     输出：[[2,2,3],[7]]
     解释：
     2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
     7 也是一个候选， 7 = 7 。
     仅有这两种组合。
     链接：https://leetcode.cn/problems/combination-sum
     */
private:
    vector<vector<int>> combinationSumRes;
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target){
        vector<int> stk;
        combinationSumCore(candidates, target, stk, 0);
        return combinationSumRes;
    }
    void combinationSumCore(vector<int>& candidates, int remain, vector<int> stk, int idx){
        if(remain == 0){
            combinationSumRes.push_back(stk);
            return;
        }
        if(remain < 0){return;}
        if(idx >= candidates.size()){return;}
        for(int i=idx; i<candidates.size(); i++){
            stk.push_back(candidates[i]);
            combinationSumCore(candidates, remain - candidates[i], stk, i);
            stk.pop_back();
        }
    }
    
    /*
     40. 组合总和 II
     给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
     candidates 中的每个数字在每个组合中只能使用 一次 。
     注意：解集不能包含重复的组合。

     输入: candidates = [10,1,2,7,6,1,5], target = 8,
     输出:
     [
     [1,1,6],
     [1,2,5],
     [1,7],
     [2,6]
     ]
     链接：https://leetcode.cn/problems/combination-sum-ii
     */
private:
    vector<vector<int>> combinationSum2res;
    vector<int> combinationSum2tmp;
    int combinationSum2CurrentSum = 0;
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        combinationSum2Backtrack(0, candidates, target);
        return combinationSum2res;
    }
    
    void combinationSum2Backtrack(int idx, vector<int>& candidates, int target){
        // idx 为start 的位置
        if(combinationSum2CurrentSum==target){
            combinationSum2res.emplace_back(combinationSum2tmp);
            return;
        }
        if(combinationSum2CurrentSum>target){return;}
        for(int i=idx; i<candidates.size(); i++){
            if(i > idx && candidates[i]==candidates[i-1]){
                continue;
            }
            else{
                // 如何在这里体现不选择 i 位置的数字？ -- 因为可以先选其它位置的，如果走到 sum == target，说明不需要该i位置的数字
                combinationSum2tmp.emplace_back(candidates[i]);
                combinationSum2CurrentSum += candidates[i];
                combinationSum2Backtrack(i+1, candidates, target);
                combinationSum2CurrentSum -= candidates[i];
                combinationSum2tmp.pop_back();
            }
        }
    }
    
    
    // 77. 组合
private:
    vector<vector<int>> combineres;
    vector<int> combinetmp;
    
public:
    vector<vector<int>> combine(int n, int k) {
        combinecore(0, k, n);
        return combineres;
    }
    void combinecore(int curr, int k, int n){
        if(combinetmp.size() == k){
            combineres.push_back(combinetmp);
            return;
        }
        if(curr >= n){
            return;
        }
        for(int i = curr; i < n; i++){
            combinetmp.push_back(i+1);
            combinecore(i+1, k, n);
            combinetmp.pop_back();
        }
        // tmp.push_back(curr);
        // core(curr+1, k, n);
        // tmp.pop_back();
        // core(curr+1, k, n);
    }
    
    /*
     216. 组合总和 III
     找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：
     只使用数字1到9,每个数字 最多使用一次
     返回 所有可能的有效组合的列表 。该列表不能包含相同的组合两次，组合可以以任何顺序返回。

     来源：力扣（LeetCode）
     链接：https://leetcode.cn/problems/combination-sum-iii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     输入: k = 3, n = 9
     输出: [[1,2,6], [1,3,5], [2,3,4]]
     解释:
     1 + 2 + 6 = 9
     1 + 3 + 5 = 9
     2 + 3 + 4 = 9
     没有其他符合的组合了。
     */
private:
    vector<vector<int>> combinationSum3res;
    vector<int> combinationSum3tmp;
    
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        combinationSum3core(0, n, k);
        return combinationSum3res;
    }
    void combinationSum3core(int idx, int remain, int k){
        // num: the current num between 1 and 9
        if(remain == 0 && combinationSum3tmp.size() == k){
            combinationSum3res.push_back(combinationSum3tmp);
            return;
        }
        if(remain < 0 || combinationSum3tmp.size() > k || idx >= 9){return;}
        // core(num+1, remain, k);
       for(int i = idx; i < 9; i++){
            combinationSum3tmp.push_back(i+1);
            combinationSum3core(i+1, remain-i-1, k);
            combinationSum3tmp.pop_back();
       }
    }
    
    // 377. 组合总和 Ⅳ
    
    int combinationSum4(vector<int>& nums, int target) {
        vector<int> dp(target + 1, 0);
        dp[0] = 1; // 什么都不选
        for(int t=1; t<=target; t++){
            for(int v: nums){
                if(v <= t &&  dp[t - v] < INT_MAX - dp[t]){
                    dp[t] += dp[t-v];
                }
            }
        }
        return dp[target];
    }
    
    // 254. 因子的组合
    vector<vector<int>> getFactors(int n) {
        vector<vector<int>> res = helper(n, 2);
        return res;
    }
    vector<vector<int>> helper(int n, int div){
        vector<vector<int>> res;
        for(int i = div; i < sqrt(n+1) ; i++){
            if(n % i == 0){
                res.push_back({i, n/i});
                for(auto tmp: helper(n/i, i)){
                    tmp.push_back(i);
                    res.push_back(tmp);
                    tmp.pop_back();
                }
            }
        }
        return res;
    }
    
    // 78. 子集
private:
    vector<vector<int>> subsetsres;
    vector<int> subsetstmp;
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        subsetsCore_1(0, nums);
        return subsetsres;
    }
    
    // 写法1
    void subsetsCore_1(int idx, vector<int>& nums){
        if(idx == nums.size()){
            subsetsres.push_back(subsetstmp);
            return;
        }
        subsetstmp.push_back(nums[idx]);
        subsetsCore_1(idx + 1, nums);
        subsetstmp.pop_back();
    }
    
    // 写法2
    void subsetsCore_2(int idx, vector<int>& nums){
        subsetsres.push_back(subsetstmp);
        for(int i=idx; i<nums.size(); i++){
            subsetstmp.push_back(nums[i]);
            subsetsCore_2(i + 1, nums);
            subsetstmp.pop_back();
            subsetsCore_2(i + 1, nums);
        }
    }
    
    /*
     46. 全排列
     给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     输入：nums = [1,2,3]
     输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
     https://leetcode.cn/problems/permutations/
     */
private:
    vector<vector<int>> premuteres;
    vector<int> premutetmp;
    unordered_map<int, bool> premuteused;
public:
    vector<vector<int>> premute(vector<int>& nums){
        premutebacktrack(nums);
        return premuteres;
    }
    
    void premutebacktrack(vector<int>& nums){
        if(premutetmp.size()==nums.size()){
            premuteres.push_back(premutetmp);
            return;
        }
        
        for(int i=0; i<nums.size(); i++){
            if(!premuteused[i]){
                premuteused[i] = true;
                premutetmp.push_back(nums[i]);
                premutebacktrack(nums);
                premutetmp.pop_back();
                premuteused[i] = false;
            }
        }
    }
    
    /*
     47. 全排列 II
     给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
     示例 1：
     输入：nums = [1,1,2]
     输出：
     [[1,1,2],
      [1,2,1],
      [2,1,1]]
     示例 2：
     输入：nums = [1,2,3]
     输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
     来源：力扣（LeetCode）
     链接：https://leetcode.cn/problems/permutations-ii
     */
private:
    vector<vector<int>> permuteUniqueRes;
    vector<int> permuteUniqueTmp;
    unordered_map<int, bool> permuteUniqueused;
    
public:
    vector<vector<int>> permuteUnique(vector<int>& nums){
        sort(nums.begin(), nums.end());
        permuteUniqueCore(nums);
        return permuteUniqueRes;
    }
    
    void permuteUniqueCore(vector<int>& nums){
        if(permuteUniqueTmp.size()==nums.size()){
            permuteUniqueRes.emplace_back(permuteUniqueTmp);
            return;
        }
        // 遍历没有使用过的元素
        for(int i=0; i<nums.size();i++){
            if(permuteUniqueused[i])
                continue;
            // 新添加的剪枝逻辑，固定相同的元素在排列中的相对位置
            if(i > 0 && nums[i] == nums[i-1] && !permuteUniqueused[i-1]){
                continue; // 如果 permuteUniqueused[i-1] == true，代表i-1已被使用过，则继续往下走。没使用过说明 i 不应该在 i-1 前面使用，continue中止i的选择。所以这里是固定了相对位置
            }
            else{
                // 选择 nums[i]
                permuteUniqueTmp.push_back(nums[i]);
                permuteUniqueused[i] = true;
                permuteUniqueCore(nums);
                permuteUniqueused[i] = false;
                permuteUniqueTmp.pop_back();
            }
        }
        
    }
    
    string getPermutation(int n, int k) {
        string res = "";
        for(int i=0; i<n; i++){
            char v = '0' + i + 1;
            res.push_back(v);
        }
        // cout << res << endl;
        k--;
        while(k){
            res = nextPermutation(res);
            k--;
        }
        return res;
    }

//    string nextPermutation(string nums) {
//        int n = nums.size();
//        int i = n - 2;
//        while(i >= 0 && nums[i] >= nums[i+1])
//            i --;
//        if(i >= 0){
//            int j = n - 1;
//            while(j >= i + 1 && nums[j] <= nums[i])
//                j--;
//            swap(nums[i], nums[j]);
//        }
//        // sort(nums.begin() + i + 1,nums.end());
//        int left = i + 1, right = n - 1;
//        while(left <= right){
//            swap(nums[left], nums[right]);
//            left++;
//            right--;
//        }
//        return nums;
//    }
    
    /*
     90. 子集 II
     给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
     解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
     输入：nums = [1,2,2]
     输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
     链接：https://leetcode.cn/problems/subsets-ii
     */
private:
    vector<vector<int>> subsetsWithDupres;
    vector<int> subsetsWithDuptmp;
    unordered_map<int, bool> subsetsWithDupused;
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        subsetsWithDupBacktrack(0, nums);
        return subsetsWithDupres;
    }
    
    void subsetsWithDupBacktrack(int idx, vector<int>& nums){
        // idx 当前的位置
        subsetsWithDupres.push_back(subsetsWithDuptmp);
        for(int i=idx; i<nums.size(); i++){
            if(i>0 && nums[i]==nums[i-1]){
                continue;
            }
            else{
                subsetsWithDuptmp.push_back(nums[i]);
                subsetsWithDupBacktrack(i+1, nums);
                subsetsWithDuptmp.pop_back();
                subsetsWithDupBacktrack(i+1, nums);
            }
        }
    }
    
    /* 752. 打开转盘锁
     你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。
     锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。
     列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。
     字符串 target 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 -1 。

     输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
     输出：6
     可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
     注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
     因为当拨动到 "0102" 时这个锁就会被锁定。

     链接：https://leetcode.cn/problems/open-the-lock
    */
private:
    string minusOne(string cur, int idx) {
        string temp = cur;
        if (temp[idx] == '0')temp[idx] = '9';
        else temp[idx] -= 1;
        return temp;
    }
    string plusOne(string cur, int idx) {
        string temp = cur;
        if (temp[idx] == '9')temp[idx] = '0';
        else temp[idx] += 1;
        return temp;
    }
    // 以下的 minusOne 是会报错的
    // string minusOne(string s, int j) {
    //     char ch[s.length()];
    //     strcpy(ch, s.c_str());
    //     if (ch[j] == '0')
    //         ch[j] = '9';
    //     else
    //         ch[j] -= 1;
    //     string res(ch);
    //     return res;
    // }

public:
    int openLock(vector<string>& deadends, string target) {  //BFS广度优先遍历
        queue<string> node;
        unordered_set<string> dead;
        unordered_set<string> visited;
        for(auto s:deadends){
            dead.insert(s);
        }
        int step = 0;
        node.push("0000");
        visited.insert("0000");
        while(!node.empty()){
            int size = node.size();
            for(int i=0;i<size;i++){
                string cur = node.front();
                node.pop();
                if(dead.find(cur)!=dead.end()) continue;
                if(cur == target) return step;
                for(int j = 0;j<4;j++){
                    string up = plusOne(cur, j);
                    if (!visited.count(up)) {
                        node.push(up);
                        visited.insert(up);
                    }
                    string down = minusOne(cur, j);
                    if (!visited.count(down)) {
                        node.push(down);
                        visited.insert(down);
                    }
                }
            }
            step++;
        }
        return -1;
    }
    
    // 6362. 最长平衡子字符串
    int findTheLongestBalancedSubstring(string s) {
       int pre = 0, cur = 0, ans = 0;
       for(int i=0; i<s.size(); i++){
           cur++; // 记录连续的0或者1个数
           if(i == s.size() - 1 || s[i] != s[i+1]){
               // 01 或者 10
               if(s[i] == '1'){
                   // 结算
                   ans = max(ans, 2 * min(cur, pre));  // 在此，pre为连续0个数，cur为连续1个数
                   pre = 0;
                   cur = 0;
               }
               else{
                   pre = cur; // 至此，cur记录的是0连续个数，用pre保留
                   cur = 0; // 切换到 1 计数
               }
           }
       }
       return ans;
   }
    /*
    704. 二分查找
    给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

    输入: nums = [-1,0,3,5,9,12], target = 9
    输出: 4
    解释: 9 出现在 nums 中并且下标为 4
     */
    int search(vector<int>& nums, int target){
        int left = 0, right = nums.size() - 1;
        while(left <= right){
            int mid = (right - left) / 2 + left;
            if(nums[mid] == target){
                return mid;
            }
            else if(nums[mid] < target){
                left = mid + 1;
            }
            else{
                right = mid - 1;
            }
        }
        return -1;
    }
    
    int left_bound(vector<int>& nums, int target){
        int left = 0, right = nums.size();
        while(left < right){
            int mid = (right - left) / 2 + left;
            if(nums[mid] == target){
                right = mid;
            }
            else if(nums[mid] > target){
                right = mid;
            }
            else{
                left = mid + 1;
                // 因为我们的「搜索区间」是 [left, right) 左闭右开，所以当 nums[mid] 被检测之后，下一步应该去 mid 的左侧或者右侧区间搜索，即 [left, mid) 或 [mid + 1, right)
            }
        }
        return left;
    }
    
    int right_bound(vector<int>& nums, int target){
        int left = 0, right = nums.size();
        while(left < right){
            int mid = (right - left) / 2 + left;
            if(nums[mid] == target){
                // 向右缩小区间
                left = mid + 1;
            }
            else if(nums[mid] < target){
                left = mid + 1;
            }
            else{
                right = mid; // 右侧是开区间，
            }
        }
        return left - 1;
    }
    
    /*
     1011. 在 D 天内送达包裹的能力
     传送带上的包裹必须在 days 天内从一个港口运送到另一个港口。
     传送带上的第i 个包裹的重量为 weights[i]。
     每一天，我们都会按给出重量（weights）的顺序往传送带上装载包裹。我们装载的重量不会超过船的最大运载重量。
     返回能在 days 天内将传送带上的所有包裹送达的船的最低运载能力。
     示例 1：
     输入：weights = [1,2,3,4,5,6,7,8,9,10], days = 5
     输出：15
     解释：
     船舶最低载重 15 就能够在 5 天内送达所有包裹，如下所示：
     第 1 天：1, 2, 3, 4, 5
     第 2 天：6, 7
     第 3 天：8
     第 4 天：9
     第 5 天：10

     来源：力扣（LeetCode）
     链接：https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     [1,2,3,4,5,6,7,8,9,10]
     [1,3,6,10,15,21,28,36,45,55]
     55/5=11 最小值
     55较大值
     (55-11)/2+11=33
     
     */
public:
    bool ship_days(vector<int>& weights, int x, int target_days){
        int days = 0;
        int tmp = 0;
        for(int i = 0; i < weights.size(); i++){
            if(weights[i] > x){
                return false;
            }
            tmp += weights[i];
            if(tmp > x){
                days += 1;
                tmp = weights[i];
            }
        }
        cout << "x" << x << "days" << days << endl;
        return (days + 1 <= target_days) ? true : false;
    }
    
    int shipWithinDays(vector<int>& weights, int days){
        int sum = 0;
        for(auto i: weights) sum+=i;
        int left = sum/days; // 最小可能值
        int right = sum + 1; // 最大可能值
        // 寻找最效左边界
        while(left < right){
            int mid = (right - left) / 2 + left;
            if(ship_days(weights, mid, days)){
                // 说明需要缩小边界
                right = mid;
            }
            else{
                left = mid + 1;
            }
            cout << mid << endl;
        }
        return left;
    }
    /*
     121. 买卖股票的最佳时机
     转移应该是：dp[天数][最大允许交易次数][是否持有 0 or 1]
     dp[i][1][0] = max(dp[i-1][1][0], dp[i-1][1][1] + prices[i])
     dp[i][1][1] = max(dp[i-1][1][1], dp[i-1][0][0] - prices[i])
              = max(dp[i-1][1][1], 0 - prices[i])
     */
    int maxProfit_121_1(vector<int>& prices){
        int n = prices.size();
        // 定义转移矩阵
        vector<vector<int>> dp(n, vector<int>(2));
        for(int i = 0; i < n; i++){
            if(i - 1 == -1){
                dp[i][1] = - prices[i];
                dp[i][0] = 0;
                continue;
            }
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = max(dp[i-1][1], -prices[i]);
        }
        return dp[n-1][0];
    }
    
    int maxProfit_121_2(vector<int>& prices){
        int n = prices.size();
        // 定义转移矩阵
        int hold = -prices[0], nohold = 0;
        for(int i = 0; i < n; i++){
            hold = max(hold, -prices[i]);
            nohold = max(nohold,  hold + prices[i]);
        }
        return nohold;
    }
    
    /*
     122. 买卖股票的最佳时机 II
     */
    int maxProfit_122_1(vector<int>& prices){
        int n = prices.size();
        vector<vector<int>> dp(n, vector<int>(2));
        for(int i = 0; i < n; i++){
            if(i - 1 == -1){
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i]);
        }
        return dp[n-1][0];
    }
    
    int maxProfit_122_2(vector<int>& prices){
        int n = prices.size();
        // 定义转移矩阵
        int hold = -prices[0], nohold = 0;
        for(int i = 0; i < n; i++){
            hold = max(hold, nohold - prices[i]);
            nohold = max(nohold,  hold + prices[i]);
        }
        return nohold;
    }
    
    /*
     123. 买卖股票的最佳时机 III
     */
    int maxProfit_123_1(vector<int>& prices){
        int n = prices.size();
        int max_k = 2;
        // dp[i][k][0 or 1]
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(max_k + 1, vector<int>(2)));
        for(int i = 0; i < n; i++){
            for(int k = max_k; k >= 1; k--){
                // k--: 剩余可交易次数逐步减少, 代表至今至多被允许进行k次交易
                // 这个疑问很正确，因为我们后文 动态规划答疑篇 有介绍 dp 数组的遍历顺序是怎么确定的，主要是根据 base case，以 base case 为起点，逐步向结果靠近。
                // 但为什么我从大到小遍历 k 也可以正确提交呢？因为你注意看，dp[i][k][..] 不会依赖 dp[i][k - 1][..]，而是依赖 dp[i - 1][k - 1][..]，而 dp[i - 1][..][..]，都是已经计算出来的，所以不管你是 k = max_k, k--，还是 k = 1, k++，都是可以得出正确答案的。
                if(i - 1 == -1){
                    dp[i][k][0] = 0;
                    dp[i][k][1] = - prices[i];
                    continue;
                }
                dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
                dp[i][k][1] = max(dp[i-1][k-1][0] - prices[i], dp[i-1][k][1]); // k-1->k，代表增加了一次交易，并且是合法的
            }
        }
        return dp[n-1][max_k][0];
    }
    
    /*
     309. 最佳买卖股票时机含冷冻期
     */
    int maxProfit_309_1(vector<int>& prices){
        int n = prices.size();
        vector<vector<int>> dp(n, vector<int>(2));
        for(int i = 0; i < n; i++){
            // base case 1
            if(i - 1 == -1){
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            // base case 2
            if(i - 2 == -1){
                dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
                dp[i][1] = max(dp[i-1][1], 0 - prices[i]);
                continue;
            }
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i]);
        }
        return dp[n-1][0];
    }
    
    int maxProfit_309_2(vector<int>& prices){
        int n = prices.size();
        // 定义转移矩阵
        int hold = -prices[0], nohold = 0;
        int nohold_pre = 0; // 代表dp[i-2][0]
        for(int i = 0; i < n; i++){
            int temp = nohold;
            hold = max(hold, nohold_pre - prices[i]);
            nohold = max(nohold, hold + prices[i]);
            nohold_pre = temp;
        }
        return nohold;
    }
    
    /*
     714. 买卖股票的最佳时机含手续费
     */
    int maxProfit_714_1(vector<int>& prices, int fee){
        int n = prices.size();
        vector<vector<int>> dp(n, vector<int>(2));
        for(int i = 0; i < n; i++){
            // base case 1
            if(i - 1 == -1){
                dp[i][0] = 0;
                dp[i][1] = -prices[i] - fee;
                continue;
            }
//            // base case 2
//            if(i - 2 == -1){
//                dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
//                dp[i][1] = max(dp[i-1][1], 0 - prices[i]);
//                continue;
//            }
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i] - fee);
        }
        return dp[n-1][0];
    }
    
    
    /*
     198. 打家劫舍
     */
    int rob198_1(vector<int>& nums){
        // dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        // base case: dp[0] = nums[0], dp[1] = max(dp[0],nums[1])
        int n = nums.size();
        vector<int> dp(n, 0);
        for(int i = 0; i < n; i++){
            if(i == 0){
                dp[i] = nums[i];
            }
            else if(i == 1){
                dp[i] = max(dp[i-1], nums[i]);
            }
            else{
                dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
            }
        }
        return dp[n-1];
    }
    
    int rob_198_2(vector<int>& nums){
        int n = nums.size();
        if(n<=2){
            return *max_element(nums.begin(), nums.end());
        }
        int first = nums[0], second = max(nums[0], nums[1]);
        for(int i = 2; i < n; i++){
            int temp = second;
            second = max(first + nums[i], second);
            first = temp;
        }
        return second;
    }
    
public:
    // 213. 打家劫舍 II
    int rob_213(vector<int>& nums) {
        int n = nums.size();
        if(n<=2){
            return *max_element(nums.begin(), nums.end());
        }
        int res1 = rob_213_help1(nums, 0, n - 1);
        int res2 = rob_213_help1(nums, 1, n);
        return max(res1, res2);
    }
private:
    int rob_213_help1(vector<int>& nums, int start, int end){
        // 左闭右开
        int first = nums[start], second = max(nums[start], nums[start+1]);
        for(int i = start + 2; i < end; i++){
            int temp = second;
            second = max(second, first + nums[i]);
            first = temp;
        }
        return second;
    }
    /*
     337. 打家劫舍 III
     */
private:
    pair<int, int> rob_337_dfs(TreeNode* cur){
        if(cur == nullptr){
            return {0,0};
        }
        // pair: 0-选择节点，1-不选择该节点
        pair<int, int> left_pair = rob_337_dfs(cur->left);
        pair<int, int> right_pair = rob_337_dfs(cur->right);
        return {
            cur->val + left_pair.second + right_pair.second,
            max(left_pair.first, left_pair.second) + max(right_pair.first, right_pair.second)
        };
    }
public:
    int rob_337(TreeNode* root){
        pair<int,int> res = rob_337_dfs(root);
        return max(res.first, res.second);
    }
    
    // 206. 反转链表
    ListNode* reverseList(ListNode* head) {
        if(head == nullptr || head->next == nullptr){
            return head;
        }
        ListNode* last = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return last;
    }
    
    // 变体：将链表的前 n 个节点反转（n <= 链表长度）
private:
    ListNode* sucessor = nullptr;
public:
    ListNode* reverseListN(ListNode* head, int n){
        if(n == 1){
            sucessor = head->next;
            return head;
        }
        ListNode* last = reverseListN(head->next, n-1);
        head->next->next = head;
        head->next = sucessor;
        return last;
    }
    
    // 25.k个一组反转链表
    ListNode* reverseKGroupHelper(ListNode* a, ListNode* b){
        // 反转 [a,b) 左闭右开的区间
        ListNode *pre, *cur, *nxt;
        pre = nullptr; cur = a; nxt = a;
        while(cur != b){
            nxt = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
    ListNode* reverseKGroup(ListNode* head, int k){
        if(head == nullptr) return head;
        // 区间 [a,b) 包含待反转元素
        ListNode *a, *b;
        a = head;
        b = head;
        // 获取b位置
        for(int i = 0; i < k; i++){
            if(b == nullptr) return head;
            b = b->next;
        }
        // 反转前 k 个元素
        ListNode* newhead = reverseKGroupHelper(a, b);
        a->next = reverseKGroup(b, k);
        return newhead;
    }
    
    // 34. 在排序数组中查找元素的第一个和最后一个位置
    bool binarySearch34(vector<int>& nums, bool lower, int target){
        int left = 0, right=nums.size()-1, ans=nums.size();
        while(left <= right){
            int mid = (right + left) >> 1;
            if(nums[mid] > target || (nums[mid] >= target && lower)){
                right = mid - 1;
                ans = mid;
            }
            else{
                left = mid + 1;
            }
        }
        return ans;
    }
    vector<int> searchRange(vector<int>& nums, int target) {
        int left_idx = binarySearch34(nums, true, target);
        int right_idx = binarySearch34(nums, false, target) - 1;
        if(left_idx <= right_idx && right_idx < nums.size() && nums[left_idx]==target && nums[right_idx]==target){
            return {left_idx, right_idx};
        }
        return {-1,-1};
    }
    
    //278. 第一个错误的版本
    int firstBadVersion(int n) {
        int left = 0, right = n-1, ans=n;
        while(left <= right){
            int mid = (right - left + 1) / 2 + left;
            if(isBadVersion(mid)){
                right = mid - 1;
                ans = mid;
            }
            else{
                left = mid + 1;
            }
        }
        return ans;
    }
    
    // 234. 判断回文链表
    bool isPalindrome_234_method1(ListNode* head){
        // 用栈存储值，再用双指针判断
        vector<int> stk;
        ListNode* tmp = head;
        while(tmp != nullptr){
            stk.emplace_back(tmp->val);
            tmp = tmp->next;
        }
        for(int i = 0; i < stk.size() / 2; i++){
            if(stk[i] != stk[stk.size() - 1 - i]){
                return false;
            }
        }
        return true;
    }
private:
    ListNode* isPalindrome_method2_left;
public:
    bool isPalindrome_234_method2(ListNode* head){
        /* 递归解法：反复调用方法并压栈，用left表示当前所在方法的左端
         类似于链表的反序列打印
         void traverse(listnode* head){
             if(head == nullptr){return;} // base case: 到了最末端
             // -->放前面为前序遍历
             traverse(head->next);
             // -->放后面为后序遍历
             cout << head->val; // 后序遍历
         }
         */
        isPalindrome_method2_left = head;
        return isPalindrome_234_method2_traverse(head);
    }
    bool isPalindrome_234_method2_traverse(ListNode* right){
        if(right == nullptr){return true;}
        bool tmp = isPalindrome_234_method2_traverse(right->next);
        bool res = tmp && (isPalindrome_method2_left->val == right->val);
        isPalindrome_method2_left = isPalindrome_method2_left->next;
        return res;
    }
    
    // 10. 正则表达式匹配
    /*
     s = ' ' + s
     p = ' ' + p
     p[j] != '*'
         if s[i]=p[j] , then dp[i][j] = dp[i-1][j-1] 直接匹配
         if s[i]!=p[j] , then dp[i][j] = false
     p[j] == '*'
         if s[i]=p[j-1], 说明p的j-1个字符，能够用于继续匹配，所以 dp[i][j] = dp[i-1][j] OR dp[i][j-2] 或者完全不匹配
         if s[i]!=p[j-1] 的情况，dp[i][j] = dp[i][j-2] 完全不匹配
     */
    bool isMatch_matches(string s, string p, int p1, int p2){
        if(p1 == 0)
            return false; // 什么都不匹配
        if(s[p1]==p[p2] or p[p2]=='.')
            return true;
        return false;
    }
    bool isMatch(string s, string p){
        // s = string, p = pattern
        s = ' ' + s;
        p = ' ' + p;
        vector<vector<bool>> dp(s.size(), vector<bool>(p.size(), false));
        dp[0][0] = true;
        
        for(int i=0; i<s.size(); i++){
            for(int j=0; j<p.size(); j++){
                if(i == 0 && j == 0){
                    continue;
                }
                else if(i > 0 && j == 0){
                    dp[i][j] = false;
                }
                else{
                    if(p[j] == '*'){
                        // 不匹配
                        dp[i][j] = dp[i][j-2] | dp[i][j];
                        if(isMatch_matches(s,p,i,j-1)){
                            // 匹配了p里面 * 前面那个字符
                            dp[i][j] = dp[i][j] | dp[i-1][j];
                        }
                    }
                    else{
                        // 没有*，需要直接匹配
                        if(isMatch_matches(s,p,i,j)){
                            dp[i][j] = dp[i-1][j-1] | dp[i][j];
                        }
                    }
                }
            }
        }
        return dp[s.size()-1][p.size()-1];
    }
    
    // 11. 盛最多水的容器
    int maxArea11(vector<int>& height){
        int left = 0, right = height.size() - 1;
        int res = INT_MIN;
        while(left <= right){
            int h = min(height[left], height[right]);
            int new_area = h*(right-left);
            res = (res > new_area)? res:new_area;
            if(height[left]<=height[right]){
                left++;
            }
            else{
                right--;
            }
        }
        return res;
    }
    
    // 238. 除自身以外数组的乘积
    
    vector<int> productExceptSelf(vector<int>& nums) {
        // 优化版
        int left = 1, right = 1, n = nums.size();
        vector<int> res(n, 1);
        for(int i=0; i<n; i++){
            // 在当下时点，left是为了i，right为了n-1-i
            res[i] *= left;
            res[n-1-i] *= right;
            left *= nums[i];
            right *= nums[n-1-i];
        }
        return res;

        // 数组版
        // vector<int> l1, l2, res;
        // l1.push_back(1);
        // l2.push_back(1);
        // int n = nums.size();
        // for(int i=0; i<n; i++){
        //     l1.push_back(l1[l1.size()-1] * nums[i]);
        //     l2.push_back(l2[l2.size()-1] * nums[n-1-i]);
        // }
        // for(int i=0; i<n; i++){
        //     res.push_back(l1[i] * l2[n-1-i]);
        // }
        // return res;
    }
    // 48. 旋转图像
    void rotate(vector<vector<int>>& matrix) {
        int h = matrix.size(), w = matrix[0].size();
        for(int i=0; i<h; i++){
            for(int j=i; j<w; j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
        for(int i=0; i<h; i++){
            for(int j=0; j<w/2; j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[i][w-1-j];
                matrix[i][w-1-j] = tmp;
            }
        }
    }
    
    // 逆序对
    vector<int> merge_sort_tmp;
    int merge_sort(vector<int>& nums, int l, int r){
        if(l >= r){return 0;}
        int mid = (l + r) / 2;
        int res = merge_sort(nums, l, mid) + merge_sort(nums, mid + 1, r);
        // begin to merge sort
        for(int p = l; p <= r; p++){
            merge_sort_tmp[p] = nums[p];
        }
        int i = l, j = mid + 1;
        for(int k = l; k <= r; k++){
            if(i == mid + 1){
                nums[k] = merge_sort_tmp[j];
                j++;
            }
            else if(j == r + 1){
                nums[k] = merge_sort_tmp[i];
                i++;
            }
            else if(merge_sort_tmp[i] <= merge_sort_tmp[j]){
                nums[k] = merge_sort_tmp[i];
                i++;
            }
            else{
                nums[k] = merge_sort_tmp[j];
                j++;
                res += (mid - i + 1);
            }
        }
        return res;
    }
    int reversePairs(vector<int>& nums) {
        for(auto i: nums){merge_sort_tmp.push_back(0);}
        return merge_sort(nums, 0, nums.size()-1);
    }
    
    // 249. 移位字符串分组
    
    string groupStringsnormalize(string s){
        if(s==""){
            return s;
        }
        string res = "a";
        int dis = s[0] - 'a';
        for(int i=1; i<s.size(); i++){
            if(s[i] - 'a' < dis){
                res.push_back(s[i] + 26 - dis);
            }
            else{
                res.push_back(s[i] - dis);
            }
        }
        return res;
    }
    vector<vector<string>> groupStrings(vector<string>& strings) {
        vector<vector<string>> res;
        unordered_map<string, vector<string>> dict;
        for(auto &s: strings){
            string curr = groupStringsnormalize(s);
            if(!dict.count(curr)){
                dict[curr] = {s};
            }
            else{
                dict[curr].push_back(s);
            }
        }
        for(auto it = dict.begin(); it != dict.end(); it++){
            res.push_back(it->second);
        }
        return res;
    }
    
};

// 重载运算符号：打印 vector<vector<int>>
ostream &operator<<(ostream &output, const vector<vector<int>> &input)
{
   for(auto vec:input){
       cout << endl;
       for(auto i: vec){
           cout << i << ',';
       }
   }
   return output;
}


// 281. 锯齿迭代器
class ZigzagIterator {
    int state = 0, res;
    vector<int>::iterator p1, p2, end1, end2;
public:
    ZigzagIterator(vector<int>& v1, vector<int>& v2) {
        p1 = v1.begin();
        p2 = v2.begin();
        end1 = v1.end();
        end2 = v2.end();
    }
    int next() {
        if((p2 != end2 && state) || (p1==end1)){
            // p1为空，或者轮到p2且p2不为空
            res = *p2;
            p2++;
        }
        else{
            res = *p1;
            p1++;
        }
        state ^= 1;
        return res;
    }
    bool hasNext() {
        if(p1 == end1 && p2 == end2){return false;}
        return true;
    }
};

#endif /* Solution_hpp */
